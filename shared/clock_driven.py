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
            slice corresponds to synapses whose delays fall into a given timestep bucket

        bucket_offsets - Integer tensor of shape [num_buckets] mapping each delay bucket
            to its corresponding timestep index. Enables indexing into a
            ring buffer at the correct location.
    """

    num_neurons = graph_config.num_neurons
    dtype = graph_config.dtype
    device = graph_config.device
    timestep = snn_config.timestep
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay

    weights = create_er_dense(config=graph_config)

    bucket_0_offset = int(min_delay / timestep)
    max_delay_steps = int(max_delay / timestep) + 1
    timestep_boundaries = torch.arange(
        start=bucket_0_offset * timestep,
        end=max_delay_steps * timestep,
        step=timestep,
        device=device,
        dtype=dtype,
    )
    delays = torch.empty(num_neurons, num_neurons, device=device, dtype=dtype)
    delays.uniform_(min_delay, max_delay)
    delay_bucket_indices = torch.bucketize(delays, timestep_boundaries) - 1
    num_buckets = max_delay_steps - bucket_0_offset

    bucketized_weights = torch.zeros(
        (num_buckets, num_neurons, num_neurons), device=device, dtype=dtype
    )
    for bucket_idx in range(num_buckets):
        mask = delay_bucket_indices == bucket_idx
        bucketized_weights[bucket_idx][mask] = weights[mask]

    # Delete the created weights tensor to free up memory
    del weights

    bucket_offsets = torch.arange(
        start=bucket_0_offset,
        end=bucket_0_offset + num_buckets,
        step=1,
        device=device,
        dtype=torch.int32,
    )

    return bucketized_weights, bucket_offsets


def create_ring_buffer(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, int]:
    """
    Create a circular buffer used to store delayed synaptic inputs

    Returns:
        ring_buffer - tensor of shape [buffer_size, num_neurons]

        buffer_size - number of discrete timesteps covered by the buffer
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

    return ring_buffer, buffer_size


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


def create_external_spike_drive(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> torch.Tensor:
    """
    Generate a fixed Poisson spike train used as external input drive

    Precomputing the spike train avoids sampling within the simulation loop.
    This makes simulations reproducible and faster.

    Returns:
        poisson_spikes - boolean tensor [num_timesteps, num_neurons]
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    num_timesteps = snn_config.num_timesteps
    poisson_prob = snn_config.poisson_prob

    # Each timestep uses independent Bernoulli sampling
    poisson_spikes = (
        torch.rand((num_timesteps, num_neurons), device=device, dtype=dtype)
        < poisson_prob
    )

    return poisson_spikes
