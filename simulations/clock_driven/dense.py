import torch

from shared.graph_creation import create_er_dense
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_dense(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """Run a clock-driven LIF spiking neural network simulation on dense graph"""

    weights = create_er_dense(config=graph_config)

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    timestep = snn_config.timestep
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    num_timesteps = snn_config.num_timesteps
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    poisson_prob = snn_config.poisson_prob
    bin_rate = snn_config.bin_rate
    num_bins = snn_config.num_bins
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay
    refractory_period = snn_config.refractory_period

    bucket_0_offset = int(min_delay / timestep)
    buffer_steps_size = int(max_delay / timestep) + 1
    timestep_boundaries = torch.arange(
        start=bucket_0_offset * timestep,
        end=buffer_steps_size * timestep,
        step=timestep,
        device=device,
        dtype=dtype,
    )
    delays = torch.empty(num_neurons, num_neurons, device=device, dtype=dtype)
    delays.uniform_(min_delay, max_delay)
    delay_buckets = torch.bucketize(delays, timestep_boundaries) - 1
    num_buckets = buffer_steps_size - bucket_0_offset

    weights_by_delay_bucket = torch.zeros(
        (num_buckets, num_neurons, num_neurons), device=device, dtype=dtype
    )
    for bucket_idx in range(num_buckets):
        mask = delay_buckets == bucket_idx
        weights_by_delay_bucket[bucket_idx][mask] = weights[mask]
    del weights

    ring_buffer = torch.zeros(
        buffer_steps_size, num_neurons, device=device, dtype=dtype
    )

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=dtype
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=dtype)
    recurrent_spikes_bool = torch.zeros(num_neurons, device=device, dtype=torch.bool)
    poisson_spikes_bool = torch.rand((num_timesteps, num_neurons)) < poisson_prob
    last_spike_times = torch.full(
        (num_neurons,), -torch.inf, device=device, dtype=dtype
    )

    spikes_per_neuron = torch.zeros(num_neurons, device=device, dtype=dtype)
    spikes_per_bin = torch.zeros(num_bins, device=device, dtype=dtype)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            current_time = t * timestep
            buffer_idx = t % buffer_steps_size
            incoming_current = ring_buffer[buffer_idx]

            synaptic_currents = synaptic_currents * synaptic_decay + incoming_current
            voltage_change = membrane_voltages - resting_voltage
            membrane_voltages = (
                voltage_change * membrane_decay + resting_voltage + synaptic_currents
            )

            can_spike_mask = current_time - last_spike_times >= refractory_period
            recurrent_spikes_bool = membrane_voltages >= threshold_voltage
            is_spiking_mask = (
                poisson_spikes_bool[t] | recurrent_spikes_bool
            ) & can_spike_mask
            combined_spikes = is_spiking_mask.to(dtype)

            # Perform full matrix multiplication then iterate over
            # buckets for better clarity when updating ring_buffer
            currents = weights_by_delay_bucket @ combined_spikes
            for bucket in range(num_buckets):
                bucket_idx_in_buffer = (
                    buffer_idx + bucket + bucket_0_offset
                ) % buffer_steps_size
                ring_buffer[bucket_idx_in_buffer] += currents[bucket]

            ring_buffer[buffer_idx].zero_()
            last_spike_times[is_spiking_mask] = current_time
            membrane_voltages[is_spiking_mask] = resting_voltage

            spikes_per_neuron += combined_spikes
            current_bin = int(current_time // bin_rate)
            spikes_per_bin[current_bin] += combined_spikes.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
