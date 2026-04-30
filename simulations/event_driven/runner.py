import torch

from shared.graph_creation import create_er_sparse
from shared.monitoring import MonitoringWindow
from shared.reporting import report_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables
from simulations.event_driven.backend import EventDrivenSimulation as Simulation


def event_driven_cpu(sim_config: SimulationConfig, seed: int) -> None:
    """Run an event-driven LIF spiking neural network simulation"""

    torch.manual_seed(seed)

    weights = create_er_sparse(sim_config)
    delays = torch.empty_like(weights.values())
    delays.uniform_(sim_config.min_delay, sim_config.max_delay)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )
    last_update_times = torch.zeros_like(last_spike_times)
    last_voltage_update_times = torch.zeros_like(last_spike_times)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    sim = Simulation(
        crow_indices=weights.crow_indices().to(torch.int32),
        col_indices=weights.col_indices().to(torch.int32),
        weights=weights.values(),
        delays=delays,
        membrane_voltages=membrane_voltages,
        synaptic_currents=synaptic_currents,
        last_update_times=last_update_times,
        last_voltage_update_times=last_voltage_update_times,
        last_spike_times=last_spike_times,
        spikes_per_neuron=spikes_per_neuron,
        spikes_per_bin=spikes_per_bin,
        max_runtime=sim_config.max_runtime,
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

    with MonitoringWindow("Simulation main") as monitor:
        result = sim.run()

    if not result["timed_out"]:
        report_statistics(
            sim_config,
            "event_driven",
            monitor.elapsed_time,
            torch.tensor(result["spikes_per_neuron"], dtype=torch.float32),
            torch.tensor(result["spikes_per_bin"], dtype=torch.float32),
        )
