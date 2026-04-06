import torch
from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables


def clock_driven_dense_cpu(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation on CPU using dense graph for weights"""

    torch.manual_seed(seed)

    resistance = sim_config.resistance
    resting_voltage = sim_config.resting_voltage
    threshold_voltage = sim_config.threshold_voltage
    membrane_decay = sim_config.membrane_decay
    synaptic_decay = sim_config.synaptic_decay
    poisson_weight = sim_config.poisson_weight
    poisson_prob = sim_config.poisson_prob
    refractory_period = sim_config.refractory_period

    bucketized_weights = build_dense_weights_bucketized_by_delay(sim_config)

    ring_buffer = create_ring_buffer(sim_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )

    random_noise, spikes_float = create_spike_tensors(sim_config)

    (
        timestep_indices,
        timestep_values,
        bin_indices,
        buffer_indices,
        bucket_indices_in_buffer,
    ) = create_lookup_tensors(sim_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    with MonitoringWindow("Simulation main"):

        for t in timestep_indices:

            current_time = timestep_values[t]
            buffer_idx = buffer_indices[t]

            random_noise.uniform_()

            synaptic_currents *= synaptic_decay
            synaptic_currents += ring_buffer[buffer_idx]
            synaptic_currents += poisson_weight * (random_noise < poisson_prob)

            outside_refractory = (current_time - last_spike_times) >= refractory_period

            alpha = synaptic_currents * resistance + resting_voltage
            new_voltages = alpha + (membrane_voltages - alpha) * membrane_decay
            membrane_voltages[outside_refractory] = new_voltages[outside_refractory]

            spikes_bool = membrane_voltages >= threshold_voltage
            spikes_float.copy_(spikes_bool)

            ring_buffer[bucket_indices_in_buffer[t]] += (
                bucketized_weights @ spikes_float
            )

            ring_buffer[buffer_idx].zero_()
            membrane_voltages[spikes_bool] = resting_voltage
            last_spike_times[spikes_bool] = current_time

            spikes_per_neuron += spikes_float
            spikes_per_bin[bin_indices[t]] += spikes_float.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
