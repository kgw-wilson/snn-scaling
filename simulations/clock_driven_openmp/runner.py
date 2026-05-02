import torch
from shared.clock_driven import (
    create_ring_buffer,
)
from shared.graph_creation import create_er_sparse
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

    weights = create_er_sparse(sim_config)
    delays = torch.empty_like(weights.values())
    delays.uniform_(sim_config.min_delay, sim_config.max_delay)
    timestep_delays = torch.floor(delays / sim_config.timestep).to(torch.int32)

    ring_buffer, buffer_size = create_ring_buffer(sim_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    sim = Simulation(
        crow_indices=weights.crow_indices().to(torch.int32),
        col_indices=weights.col_indices().to(torch.int32),
        weights=weights.values(),
        timestep_delays=timestep_delays,
        random_noise=torch.empty_like(membrane_voltages),
        membrane_voltages=membrane_voltages,
        synaptic_currents=synaptic_currents,
        last_spike_times=last_spike_times,
        ring_buffer=ring_buffer,
        spikes_per_neuron=spikes_per_neuron,
        spikes_per_bin=spikes_per_bin,
        max_runtime=sim_config.max_runtime,
        num_neurons=sim_config.num_neurons,
        num_timesteps=sim_config.num_timesteps,
        num_bins=sim_config.num_bins,
        buffer_size=buffer_size,
        timesteps_per_bin=sim_config.timesteps_per_bin,
        timestep=sim_config.timestep,
        resistance=sim_config.resistance,
        poisson_weight=sim_config.poisson_weight,
        poisson_prob=sim_config.poisson_prob,
        refractory_period=sim_config.refractory_period,
        membrane_decay=sim_config.membrane_decay,
        synaptic_decay=sim_config.synaptic_decay,
        resting_voltage=sim_config.resting_voltage,
        threshold_voltage=sim_config.threshold_voltage,
    )

    with MonitoringWindow("Simulation main") as monitor:
        result = sim.run()

    if not result["timed_out"]:
        report_statistics(
            sim_config,
            "clock_driven_openmp",
            monitor.elapsed_time,
            torch.tensor(result["spikes_per_neuron"], dtype=torch.float32),
            torch.tensor(result["spikes_per_bin"], dtype=torch.float32),
        )
