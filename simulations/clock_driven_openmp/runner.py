import numpy as np
import torch
from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables
from simulations.clock_driven_openmp.backend import (
    ClockDrivenOpenmpSimulation as Simulation,
)


def clock_driven_openmp(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation on CPU using dense graph for weights"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    timestep = sim_config.timestep
    timesteps_per_bin = sim_config.timesteps_per_bin
    resistance = sim_config.resistance
    resting_voltage = sim_config.resting_voltage
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

    sim = Simulation(
        bucketized_weights.numpy().astype(np.float32),
        membrane_voltages.numpy().astype(np.float32),
        synaptic_currents.numpy().astype(np.float32),
        last_spike_times.numpy().astype(np.float32),
        spikes_per_neuron.numpy().astype(np.int32),
        spikes_per_bin.numpy().astype(np.int32),
        sim_config.num_timesteps,
        sim_config.timestep,
        sim_config.resistance,
        sim_config.poisson_prob,
        sim_config.poisson_weight,
        sim_config.refractory_period,
        sim_config.membrane_decay,
        sim_config.synaptic_decay,
        sim_config.resting_voltage,
        sim_config.threshold_voltage,
        seed,
    )

    with MonitoringWindow("Simulation main"):
        result = sim.run()

    # report_spike_statistics(
    #     torch.tensor(result["spikes_per_neuron"], torch.float32),
    #     torch.tensor(result["spikes_per_bin"], torch.float32),
    # )
    print("Done")
