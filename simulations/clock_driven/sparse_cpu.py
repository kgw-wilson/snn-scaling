import numpy as np
import torch
import sparse_dot_mkl
from shared.clock_driven import (
    build_sparse_weights_bucketized_by_delay,
    create_ring_buffer,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.simulation_config import SimulationConfig
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.utils import create_state_variables


def clock_driven_sparse_cpu(sim_config: SimulationConfig, seed: int):
    """Run clock-driven SNN simulation on CPU using sparse csr matrix for weights"""

    torch.manual_seed(seed)

    num_neurons = sim_config.num_neurons
    resistance = sim_config.resistance
    resting_voltage = sim_config.resting_voltage
    threshold_voltage = sim_config.threshold_voltage
    membrane_decay = sim_config.membrane_decay
    synaptic_decay = sim_config.synaptic_decay
    poisson_weight = sim_config.poisson_weight
    poisson_prob = sim_config.poisson_prob
    refractory_period = sim_config.refractory_period

    bucketized_weights, num_buckets = build_sparse_weights_bucketized_by_delay(
        sim_config, use_numpy=True
    )

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

    # Convert everything to numpy even if it doesn't need to be to keep everything
    # aligned. Intentionally does not use .cpu() to throw error upon device mismatch.
    # Random noise has to be float64 for compatibility with numpy's default_rng.random
    ring_buffer = ring_buffer.numpy()
    membrane_voltages = membrane_voltages.numpy()
    synaptic_currents = synaptic_currents.numpy()
    last_spike_times = last_spike_times.numpy()
    random_noise = random_noise.to(torch.float64).numpy()
    spikes_float = spikes_float.numpy()
    timestep_indices = timestep_indices.numpy()
    timestep_values = timestep_values.numpy()
    bin_indices = bin_indices.numpy()
    buffer_indices = buffer_indices.numpy()
    bucket_indices_in_buffer = bucket_indices_in_buffer.numpy()

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    rng = np.random.default_rng(seed)

    with MonitoringWindow("Simulation main"):

        for t in timestep_indices:

            current_time = timestep_values[t]
            buffer_idx = buffer_indices[t]

            rng.random(size=num_neurons, out=random_noise)

            synaptic_currents *= synaptic_decay
            synaptic_currents += ring_buffer[buffer_idx]
            synaptic_currents += poisson_weight * (random_noise < poisson_prob)

            outside_refractory = current_time - last_spike_times >= refractory_period

            alpha = synaptic_currents * resistance + resting_voltage
            new_voltages = alpha + (membrane_voltages - alpha) * membrane_decay
            membrane_voltages[outside_refractory] = new_voltages[outside_refractory]

            spikes_bool = membrane_voltages >= threshold_voltage
            np.copyto(spikes_float, spikes_bool)

            for bucket_idx in range(num_buckets):
                target_idx = bucket_indices_in_buffer[t][bucket_idx]
                ring_buffer[target_idx] += sparse_dot_mkl.dot_product_mkl(
                    bucketized_weights[bucket_idx], spikes_float
                )

            ring_buffer[buffer_idx].fill(0.0)
            membrane_voltages[spikes_bool] = resting_voltage
            last_spike_times[spikes_bool] = current_time

            spikes_per_neuron += spikes_float
            spikes_per_bin[bin_indices[t]] += spikes_float.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
