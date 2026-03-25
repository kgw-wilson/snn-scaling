from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_state_variables,
    create_external_spike_drive,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_dense(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """Run a clock-driven LIF spiking neural network simulation on dense graph"""

    # Unpack config objects to avoid attribute lookups in the main simulation loop
    dtype = graph_config.dtype
    timestep = snn_config.timestep
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    num_timesteps = snn_config.num_timesteps
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    bin_rate = snn_config.bin_rate
    refractory_period = snn_config.refractory_period

    bucketized_weights, bucket_offsets = build_dense_weights_bucketized_by_delay(
        graph_config=graph_config, snn_config=snn_config
    )

    ring_buffer, buffer_size = create_ring_buffer(
        graph_config=graph_config, snn_config=snn_config
    )

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config=graph_config, snn_config=snn_config
    )
    poisson_spikes = create_external_spike_drive(
        graph_config=graph_config, snn_config=snn_config
    )

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config=graph_config, snn_config=snn_config
    )

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            # time and indices
            current_time = t * timestep
            buffer_idx = t % buffer_size

            # current update
            incoming_current = ring_buffer[buffer_idx]
            synaptic_currents = synaptic_currents * synaptic_decay + incoming_current

            # voltage update
            membrane_voltages = (
                (membrane_voltages - resting_voltage) * membrane_decay
                + resting_voltage
                + synaptic_currents
            )

            # spike generation
            can_spike_mask = current_time - last_spike_times >= refractory_period
            recurrent_spikes = membrane_voltages >= threshold_voltage
            is_spiking_mask = (poisson_spikes[t] | recurrent_spikes) & can_spike_mask
            spike_vector = is_spiking_mask.to(dtype)

            # propagation (schedule future current in buffer)
            bucket_indices_in_buffer = (buffer_idx + bucket_offsets) % buffer_size
            ring_buffer[bucket_indices_in_buffer] += bucketized_weights @ spike_vector

            # variable resets
            ring_buffer[buffer_idx].zero_()
            last_spike_times[is_spiking_mask] = current_time
            membrane_voltages[is_spiking_mask] = resting_voltage

            # reporting
            spikes_per_neuron += spike_vector
            spikes_per_bin[int(current_time // bin_rate)] += spike_vector.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
