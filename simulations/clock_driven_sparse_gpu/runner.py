import torch
from shared.clock_driven import (
    build_sparse_weights_bucketized_by_delay,
    create_ring_buffer,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables
from simulations.clock_driven_sparse_gpu.backend import (
    ClockDrivenSparseGpuSimulation as Simulation,
)


def clock_driven_sparse_gpu(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation on GPU using sparse csr tensor for weights"""

    torch.manual_seed(seed)

    bucketized_weights, num_buckets = build_sparse_weights_bucketized_by_delay(
        sim_config
    )

    ring_buffer, buffer_size = create_ring_buffer(sim_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )

    (
        _,
        _,
        bucket_indices_in_buffer,
    ) = create_lookup_tensors(sim_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    sim = Simulation(
        bucket_indices_in_buffer=bucket_indices_in_buffer,
        bucketized_weights=bucketized_weights,
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
        num_buckets=num_buckets,
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
            "clock_driven_sparse_gpu",
            monitor.elapsed_time,
            torch.tensor(result["spikes_per_neuron"], dtype=torch.float32),
            torch.tensor(result["spikes_per_bin"], dtype=torch.float32),
        )
