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


def clock_driven_dense_gpu(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation on GPU using dense graph for weights"""

    torch.manual_seed(seed)

    timestep = sim_config.timestep
    timesteps_per_bin = sim_config.timesteps_per_bin
    resting_voltage = sim_config.resting_voltage
    resistance = sim_config.resistance
    threshold_voltage = sim_config.threshold_voltage
    membrane_decay = sim_config.membrane_decay
    synaptic_decay = sim_config.synaptic_decay
    poisson_weight = sim_config.poisson_weight
    poisson_prob = sim_config.poisson_prob
    refractory_period = sim_config.refractory_period

    bucketized_weights = build_dense_weights_bucketized_by_delay(sim_config)

    ring_buffer, buffer_size = create_ring_buffer(sim_config)

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

            current_time = t * timestep
            buffer_idx = (buffer_idx + 1) % buffer_size
            bin_idx = t // timesteps_per_bin

            random_noise.uniform_()

            synaptic_currents *= synaptic_decay
            synaptic_currents += ring_buffer[buffer_idx]
            synaptic_currents += poisson_weight * (random_noise < poisson_prob)

            outside_refractory = current_time - last_spike_times >= refractory_period

            alpha = synaptic_currents * resistance + resting_voltage
            new_voltages = alpha + (membrane_voltages - alpha) * membrane_decay
            membrane_voltages = torch.where(
                outside_refractory, new_voltages, membrane_voltages
            )

            spikes_bool = membrane_voltages >= threshold_voltage
            spikes_float.copy_(spikes_bool)

            ring_buffer.index_add_(
                0, bucket_indices_in_buffer[t], bucketized_weights @ spikes_float
            )

            ring_buffer[buffer_idx].zero_()
            membrane_voltages = torch.where(
                spikes_bool, resting_voltage, membrane_voltages
            )
            last_spike_times = torch.where(spikes_bool, current_time, last_spike_times)

            spikes_per_neuron += spikes_bool
            spikes_per_bin[bin_idx] += spikes_bool.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
