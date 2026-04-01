from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_state_variables,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig


def clock_driven_dense_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """Run a clock-driven LIF spiking neural network simulation on dense graph"""

    # Unpack config to avoid attribute lookups in simulation loop
    resting_voltage = snn_config.resting_voltage
    membrane_bias = snn_config.membrane_bias
    threshold_voltage = snn_config.threshold_voltage
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    poisson_prob = snn_config.poisson_prob
    refractory_period = snn_config.refractory_period

    bucketized_weights = build_dense_weights_bucketized_by_delay(
        graph_config, snn_config
    )

    ring_buffer = create_ring_buffer(graph_config, snn_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config, snn_config
    )

    random_noise, spikes_float = create_spike_tensors(graph_config=graph_config)

    (
        timesteps,
        timestep_values,
        bin_indices,
        buffer_indices,
        bucket_indices_in_buffer,
    ) = create_lookup_tensors(graph_config, snn_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config, snn_config
    )

    with MonitoringWindow("simulation main loop"):

        for t in timesteps:

            current_time = timestep_values[t]
            buffer_idx = buffer_indices[t]

            random_noise.uniform_()

            synaptic_currents *= synaptic_decay
            synaptic_currents += ring_buffer[buffer_idx]

            membrane_voltages *= membrane_decay
            membrane_voltages += membrane_bias
            membrane_voltages += synaptic_currents

            poisson_spikes = random_noise < poisson_prob
            can_spike_mask = current_time - last_spike_times >= refractory_period
            recurrent_spikes = membrane_voltages >= threshold_voltage
            spikes_bool = (poisson_spikes | recurrent_spikes) & (can_spike_mask)
            spikes_float.copy_(spikes_bool)

            ring_buffer[bucket_indices_in_buffer[t]] += (
                bucketized_weights @ spikes_float
            )

            ring_buffer[buffer_idx].zero_()
            membrane_voltages[spikes_bool] = resting_voltage
            last_spike_times[spikes_bool] = current_time

            spikes_per_neuron += spikes_bool
            spikes_per_bin[bin_indices[t]] += spikes_float.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
