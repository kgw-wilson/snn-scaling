import numpy as np
import torch

from shared.graph_creation import create_er_sparse
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables
import simulations.event_driven_cpu_cpp as cpu_cpp


def event_driven_cpu(sim_config: SimulationConfig, seed: int) -> None:
    """Run an event-driven LIF spiking neural network simulation"""

    np.random.seed(seed)

    weights = create_er_sparse(sim_config, use_numpy=True)

    delays = np.random.uniform(
        low=sim_config.min_delay, high=sim_config.max_delay, size=len(weights.data)
    ).astype(np.float32)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )
    last_update_times = np.full(sim_config.num_neurons, 0.0, dtype=np.float32)
    last_voltage_update_times = np.full(
        sim_config.num_neurons, -sim_config.timestep, dtype=np.float32
    )
    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    sim = cpu_cpp.Simulation(
        indptr=weights.indptr.astype(np.int32),
        indices=weights.indices.astype(np.int32),
        weights=weights.data.astype(np.float32),
        delays=delays,
        membrane_voltages=membrane_voltages.numpy().astype(np.float32),
        synaptic_currents=synaptic_currents.numpy().astype(np.float32),
        last_update_times=last_update_times,
        last_voltage_update_times=last_voltage_update_times,
        last_spike_times=last_spike_times.numpy().astype(np.float32),
        spikes_per_neuron=spikes_per_neuron.numpy().astype(np.int32),
        spikes_per_bin=spikes_per_bin.numpy().astype(np.int32),
        num_neurons=sim_config.num_neurons,
        num_bins=sim_config.num_bins,
        resistance=sim_config.resistance,
        simulation_time=sim_config.simulation_time,
        poisson_rate=sim_config.poisson_rate,
        poisson_weight=sim_config.poisson_weight,
        refractory_period=sim_config.refractory_period,
        membrane_time_constant=sim_config.membrane_time_constant,
        synaptic_time_constant=sim_config.synaptic_time_constant,
        resting_voltage=sim_config.resting_voltage,
        threshold_voltage=sim_config.threshold_voltage,
        bin_rate=sim_config.bin_rate,
        seed=seed,
    )

    with MonitoringWindow("Simulation main"):
        result = sim.run()

    report_spike_statistics(
        torch.tensor(result["spikes_per_neuron"], dtype=sim_config.dtype),
        torch.tensor(result["spikes_per_bin"], dtype=sim_config.dtype),
    )
