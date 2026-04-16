from scipy.sparse import csr_matrix
import torch
from shared.graph_creation import create_er_dense
from shared.simulation_config import SimulationConfig


def build_dense_weights_bucketized_by_delay(
    sim_config: SimulationConfig,
) -> torch.Tensor:
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
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device

    weights = create_er_dense(sim_config)

    delay_bucket_indices, num_buckets = _compute_delay_buckets(sim_config)

    bucketized_weights = torch.zeros(
        (num_buckets, num_neurons, num_neurons), device=device, dtype=torch.float32
    )
    for bucket_idx in range(num_buckets):
        mask = delay_bucket_indices == bucket_idx
        bucketized_weights[bucket_idx][mask] = weights[mask]

    return bucketized_weights


def build_sparse_weights_bucketized_by_delay(
    sim_config: SimulationConfig, use_numpy: bool
) -> tuple[list[torch.Tensor] | list[csr_matrix], int]:
    """
    Build a list of sparse weight tensors/np.ndarrays organized into discrete delay buckets

    Relies on a dense weight graph for simplicity, but is very memory inefficient because of
    its creation of the dense weight graph and multiple sparse matrices.

    TODO: update to make more memory efficient.

    Returns:
        bucketized_weights: list of either scipy.sparse.csr_matrix or torch.sparse_csr_tensor
            depending on use_numpy value. CSR matrix of shape [num_neurons, num_neurons]
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device

    if use_numpy and device != torch.device("cpu"):
        raise ValueError("This function should only use numpy when running on CPU.")

    weights = create_er_dense(sim_config)

    delay_bucket_indices, num_buckets = _compute_delay_buckets(sim_config)

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
                dtype=torch.float32,
            )
            bucketized_weights.append(weights_coo.coalesce().to_sparse_csr())

    return bucketized_weights, num_buckets


def create_ring_buffer(sim_config: SimulationConfig) -> tuple[torch.Tensor, int]:
    """
    Create a circular buffer used to store delayed synaptic inputs

    Returns:
        ring_buffer - tensor of shape [buffer_size, num_neurons]

        buffer_size - int
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    max_delay = sim_config.max_delay
    timestep = sim_config.timestep

    buffer_size = int(max_delay / timestep) + 1
    ring_buffer = torch.zeros(
        buffer_size,
        num_neurons,
        device=device,
        dtype=torch.float32,
    )

    return ring_buffer, buffer_size


def create_spike_tensors(sim_config: SimulationConfig) -> torch.Tensor:
    """
    Returns per-timestep spike tensors to avoid re-allocation in loop

    Random noise is allocated once here and should be populated in-place
    using .uniform_() and then used to generate the external spikes for a
    timestep by comparing to poisson_prob. spikes_float is allocated here
    and should be updated by copying spikes_bool into it because that
    avoids allocating new tensors with spikes_bool.to(torch.float32) within
    the simulation loop.

    Returns:
        random_noise - empty float tensor [num_neurons]

        spikes_float - empty float tensor [num_neurons]
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device

    random_noise = torch.empty(num_neurons, device=device, dtype=torch.float32)
    spikes_float = torch.empty(num_neurons, device=device, dtype=torch.float32)

    return random_noise, spikes_float


def create_lookup_tensors(
    sim_config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tensors used to quickly lookup values based on timestep index

    In the case of timestep values, bin indices, and buffer indices, these are computed
    on the fly in the simulation loop using simple computation on the timestep index
    because that is probably faster than memory access (and allocates less memory).

    Returns:
        timestep_indices - int tensor [num_timesteps] stores each timestep index

        buffer_index - 0-dim int tensor stores initial value of buffer index

        bucket_indices_in_buffer - [num_timesteps, num_buckets] maps a timestep
            to offsets to correctly schedule generated current into the ring buffer
    """

    device = sim_config.device
    num_timesteps = sim_config.num_timesteps
    timestep = sim_config.timestep
    max_delay = sim_config.max_delay
    min_delay = sim_config.min_delay

    buffer_size = int(max_delay / timestep) + 1
    min_delay_steps = int(min_delay / timestep)

    timestep_indices = torch.arange(0, num_timesteps, device=device)

    buffer_index = torch.tensor(0, dtype=torch.int32, device=device)

    bucket_offsets = torch.arange(min_delay_steps, buffer_size, device=device)
    bucket_indices_in_buffer = (
        timestep_indices.unsqueeze(1) + bucket_offsets.unsqueeze(0)
    ) % buffer_size

    return (
        timestep_indices,
        buffer_index,
        bucket_indices_in_buffer,
    )


def _compute_delay_buckets(sim_config: SimulationConfig) -> tuple[torch.Tensor, int]:
    """
    Compute delay bucket indices and metadata shared across dense and sparse builds

    Delays are sampled from uniform distribution. 

    Subtract 1 because torch.bucketize returns indices in the range
    [0, len(boundaries)], where index 0 corresponds to values strictly less
    than the first boundary. Since valid delays start at or above
    bucket_0_offset * timestep, the resulting indices will always be >= 1
    before shifting.

    No clamping is required because delays are guaranteed to be within the
    valid range [bucket_0_offset * timestep, max_delay_steps * timestep),
    so they will never fall outside the defined bucket boundaries.

    `right=True` ensures that values exactly equal to a boundary are placed
    in the correct bucket. In the special case where
    `min_delay == max_delay == timestep`, all values correctly map to bucket 0.

    Returns:
        delay_bucket_indices: float tensor [num_neurons, num_neurons] mapping
            each connection to a delay bucket

        num_buckets: Total number of delay buckets
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    timestep = sim_config.timestep
    min_delay = sim_config.min_delay
    max_delay = sim_config.max_delay

    bucket_0_offset = int(min_delay / timestep)
    max_delay_steps = int(max_delay / timestep) + 1

    timestep_boundaries = torch.arange(
        start=bucket_0_offset * timestep,
        end=max_delay_steps * timestep,
        step=timestep,
        device=device,
    )

    delays = torch.empty(num_neurons, num_neurons, device=device, dtype=torch.float32)
    delays.uniform_(min_delay, max_delay)

    delay_bucket_indices = torch.bucketize(delays, timestep_boundaries, right=True) - 1

    num_buckets = max_delay_steps - bucket_0_offset

    return delay_bucket_indices, num_buckets
