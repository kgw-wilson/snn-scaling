from scipy.sparse import csr_matrix
import torch
from shared.graph_creation import create_er_dense
from shared.simulation_config import ERGraphConfig, SNNConfig


def build_dense_weights_bucketized_by_delay(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dense weight tensor organized into discrete delay buckets

    This function creates a fully-connected weight matrix then converts it
    into a 3D tensor indexed by discrete synaptic delays. Delays are uniformly
    sampled per connection. Memory footprint of weights is
    O(num_buckets * num_neurons^2). Thus, increasing num_buckets increases
    precision but linearly increases memory.

    Returns:
        bucketized_weights - Tensor of shape [num_buckets, num_neurons, num_neurons], where each
            slice corresponds to synapses whose delays fall into a given timestep bucket.

        bucket_offsets - Integer tensor of shape [num_buckets] mapping each delay bucket
            to its corresponding timestep index. Enables indexing into a
            ring buffer at the correct location.
    """

    num_neurons = graph_config.num_neurons
    dtype = graph_config.dtype
    device = graph_config.device

    weights = create_er_dense(config=graph_config)

    delay_bucket_indices, bucket_offsets, num_buckets = _compute_delay_buckets(
        graph_config=graph_config, snn_config=snn_config
    )

    bucketized_weights = torch.zeros(
        (num_buckets, num_neurons, num_neurons), device=device, dtype=dtype
    )
    for bucket_idx in range(num_buckets):
        mask = delay_bucket_indices == bucket_idx
        bucketized_weights[bucket_idx][mask] = weights[mask]

    return bucketized_weights, bucket_offsets


def build_sparse_weights_bucketized_by_delay(
    graph_config: ERGraphConfig, snn_config: SNNConfig, use_numpy: bool
) -> tuple[list, int]:
    """
    Build a list of sparse weight tensors/np.ndarrays organized into discrete delay buckets

    Relies on a dense weight graph for simplicity, but is very memory inefficient because of
    its creation of

    TODO: update to make more memory efficient.

    Returns:
        bucketized_weights: list of either scipy.sparse.csr_matrix or torch.sparse_csr_tensor
            depending on use_numpy value. CSR matrix of shape [num_neurons, num_neurons]

        bucket_offsets - Integer tensor of shape [num_buckets] mapping each delay bucket
            to its corresponding timestep index. Enables indexing into a
            ring buffer at the correct location.
    """

    num_neurons = graph_config.num_neurons
    dtype = graph_config.dtype
    device = graph_config.device

    if use_numpy and device != torch.device("cpu"):
        raise ValueError("This function should only use numpy when running on CPU.")

    weights = create_er_dense(config=graph_config)

    delay_bucket_indices, bucket_offsets, num_buckets = _compute_delay_buckets(
        graph_config=graph_config, snn_config=snn_config
    )

    row_idx, col_idx = torch.nonzero(weights, as_tuple=True)
    values = weights[row_idx, col_idx]
    edge_buckets = delay_bucket_indices[row_idx, col_idx]

    if use_numpy:
        row_idx = row_idx.numpy()
        col_idx = col_idx.numpy()
        values = values.numpy()
        edge_buckets = edge_buckets.numpy()

    bucketized_weights = []

    for bucket_idx in range(num_buckets):

        mask = edge_buckets == bucket_idx

        rows = row_idx[mask]
        cols = col_idx[mask]
        vals = values[mask]

        if use_numpy:
            bucketized_weights.append(
                csr_matrix((vals, (rows, cols)), shape=(num_neurons, num_neurons))
            )
        else:
            indices = torch.stack([rows, cols])
            weights_coo = torch.sparse_coo_tensor(
                indices,
                vals,
                size=(num_neurons, num_neurons),
                device=device,
                dtype=dtype,
            )
            bucketized_weights.append(weights_coo.coalesce().to_sparse_csr())

    return bucketized_weights, bucket_offsets


def create_ring_buffer(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, int]:
    """
    Create a circular buffer used to store delayed synaptic inputs

    Returns:
        ring_buffer - tensor of shape [buffer_size, num_neurons]
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype

    max_delay = snn_config.max_delay
    timestep = snn_config.timestep

    buffer_size = int(max_delay / timestep) + 1
    ring_buffer = torch.zeros(
        buffer_size,
        num_neurons,
        device=device,
        dtype=dtype,
    )

    return ring_buffer


def create_state_variables(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize dynamic neuron state variables for the LIF simulation

    Returns:
        membrane_voltages - membrane potential per neuron

        synaptic_currents - synaptic input per neuron

        last_spike_times - timestamp of last spike per neuron (for refractory enforcement)
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    resting_voltage = snn_config.resting_voltage

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=dtype
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=dtype)

    # Initialize to -inf to allow immediate firing
    last_spike_times = torch.full(
        (num_neurons,), -torch.inf, device=device, dtype=dtype
    )

    return membrane_voltages, synaptic_currents, last_spike_times


def create_spike_tensors(graph_config: ERGraphConfig) -> torch.Tensor:
    """
    Returns per-timestep spike tensors to avoid re-allocation in loop

    Random noise is allocated once here and should be populated in-place
    using .uniform_() and then used to generate the external spikes for a
    timestep by comparing to poisson_prob. spikes_float is allocated here
    and should be updated with spikes_float[:] = spikes_bool because that
    avoids allocating new tensors with spikes_bool.to(dtype) within the
    simulation loop.

    Returns:
        random_noise - empty tensor [num_neurons] with dtype from graph_config

        spikes_float - empty tensor [num_neurons] with dtype from graph_config.
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype

    random_noise = torch.empty(num_neurons, device=device, dtype=dtype)
    spikes_float = torch.empty(num_neurons, device=device, dtype=dtype)

    return random_noise, spikes_float


def create_lookup_tensors(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tensors used to lookup timestep value or bin index based on timestep index

    These tensors help speed up compiled PyTorch code compared to passing in
    unique Python scalars for each timestep.

    Returns:
        timestep_values - tensor [num_timesteps] with dtype from graph_config

        bin_indices - int tensor [num_timesteps]
    """

    dtype = graph_config.dtype
    device = graph_config.device
    num_timesteps = snn_config.num_timesteps
    timestep = snn_config.timestep
    timesteps_per_bin = snn_config.timesteps_per_bin

    timestep_values = (
        torch.arange(0, num_timesteps, device=device, dtype=dtype) * timestep
    )
    bin_indices = torch.arange(0, num_timesteps, device=device) // timesteps_per_bin

    return timestep_values, bin_indices


def _compute_delay_buckets(
    graph_config: ERGraphConfig,
    snn_config: SNNConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute delay bucket indices and metadata shared across dense and sparse builds

    Returns:
        delay_bucket_indices: torch.Tensor of shape [num_neurons, num_neurons] mapping
            each connection to a delay bucket

        bucket_offsets - Integer tensor of shape [num_buckets] mapping each delay bucket
            to its corresponding timestep index. Enables indexing into a
            ring buffer at the correct location.

        num_buckets: Total number of delay buckets
    """

    num_neurons = graph_config.num_neurons
    dtype = graph_config.dtype
    device = graph_config.device

    timestep = snn_config.timestep
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay

    bucket_0_offset = int(min_delay / timestep)
    max_delay_steps = int(max_delay / timestep) + 1

    timestep_boundaries = torch.arange(
        start=bucket_0_offset * timestep,
        end=max_delay_steps * timestep,
        step=timestep,
        device=device,
    )

    # Sample delays per connection using a uniform distribution
    delays = torch.empty(num_neurons, num_neurons, device=device, dtype=dtype)
    delays.uniform_(min_delay, max_delay)

    # Subtract by 1 because bucket values start at 1. Delays will never be
    # lower than bucket_0_offset * timestep and thus will always be at least at
    # bucket 1. Similarly, no need to clamp because no delay will be greater
    # than or equal to max_delay_steps * timestep.
    delay_bucket_indices = torch.bucketize(delays, timestep_boundaries) - 1

    num_buckets = max_delay_steps - bucket_0_offset

    bucket_offsets = torch.arange(
        start=bucket_0_offset,
        end=bucket_0_offset + num_buckets,
        device=device,
    )

    return delay_bucket_indices, bucket_offsets, num_buckets
