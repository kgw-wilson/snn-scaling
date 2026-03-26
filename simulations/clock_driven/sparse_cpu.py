from shared.clock_driven import build_sparse_weights_bucketized_by_delay, create_ring_buffer, create_state_variables, create_external_spike_drive
from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors


def run_simulation_sparse_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run SNN simulation using sparse csr matrix for synaptic weights on CPU"""

    # Unpack config objects to avoid attribute lookups
    timestep = snn_config.timestep
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    num_timesteps = snn_config.num_timesteps
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    bin_rate = snn_config.bin_rate
    refractory_period = snn_config.refractory_period


    bucketized_weights, bucket_offsets = build_sparse_weights_bucketized_by_delay(
        graph_config=graph_config, snn_config=snn_config, use_numpy=True
    )
    bucket_0_offset = bucket_offsets[0]

    ring_buffer, buffer_size = create_ring_buffer(
        graph_config=graph_config, snn_config=snn_config
    )

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config=graph_config, snn_config=snn_config
    )
    poisson_spikes = create_external_spike_drive(
        graph_config=graph_config, snn_config=snn_config
    )

    # Intentionally do not use .cpu() to throw error upon device mismatch
    ring_buffer = ring_buffer.numpy()
    membrane_voltages = membrane_voltages.numpy()
    synaptic_currents = synaptic_currents.numpy()
    last_spike_times = last_spike_times.numpy()
    poisson_spikes = poisson_spikes.numpy()

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
            spikes = (poisson_spikes[t] | recurrent_spikes) & can_spike_mask

            # propagation (schedule future current in buffer)
            for bucket_idx in range(len(bucketized_weights)):
                bucket_idx_in_buffer = (buffer_idx + bucket_0_offset + bucket_idx) % buffer_size
                ring_buffer[bucket_idx_in_buffer] = bucketized_weights[bucket_idx].dot(spikes)

            # variable resets
            ring_buffer[buffer_idx].fill(0.0)
            last_spike_times[spikes] = current_time
            membrane_voltages[spikes] = resting_voltage

            # reporting
            spikes_per_neuron += spikes
            spikes_per_bin[int(current_time // bin_rate)] += spikes.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
