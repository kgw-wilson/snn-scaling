"""
Run a single simulation in this process

run_all.py launches one process per simulation so the peak memory reported in
results.csv belongs to exactly one simulation.

Usage: python -m simulations.run_one <simulation> <device> <num_neurons> <connection_prob> <repeat> <num_threads>
"""

import math
import sys

import torch

from shared.reporting import record_memory_baseline
from shared.simulation_config import SimulationConfig
from shared.utils import get_deterministic_seed
from simulations.clock_driven_dense.runner import clock_driven_dense
from simulations.clock_driven_openmp.runner import clock_driven_openmp
from simulations.clock_driven_sparse.runner import clock_driven_sparse
from simulations.event_driven.runner import event_driven_cpu

TIMED_OUT_EXIT_CODE = 3

SIMULATIONS = {
    simulation.__name__: simulation
    for simulation in (
        clock_driven_dense,
        clock_driven_sparse,
        clock_driven_openmp,
        event_driven_cpu,
    )
}


def build_sim_config(
    device: str, num_neurons: int, connection_prob: float
) -> SimulationConfig:
    # Set resting voltage to 0 for simplicity. Resistance * Capacitance should be around 5-20 ms.
    # The values for recurrent_weight and poisson_weight are experimental but are linearly
    # dependent on resting_voltage and inversely dependent on resistance for basic consistency
    # with I = V/R. They depend inversely on the square root of the number of connections to keep
    # variance roughly the same through the network. Finally, poisson_weight is scaled by the number
    # of connections since poisson current only affects one neuron while recurrent current affects
    # all connected neurons. Initial experiments with these weights produced total poisson current
    # about 10-20% of total recurrent current which is reasonable.
    # min_delay should be greater than or equal to timestep, and refractory period is a multiple
    # of timestep so neurons will not fire again in the same timestep which is crucial for correctness
    # in the event-driven simulation.
    return SimulationConfig(
        max_runtime=5 * 60,
        num_neurons=num_neurons,
        connection_prob=connection_prob,
        device_str=device,
        timestep=1e-3,
        simulation_time=1e-3 * 1000,
        resistance=10.0,
        capacitance=1e-3,
        synaptic_time_constant=5e-3,
        resting_voltage=0.0,
        threshold_voltage=20e-3,
        recurrent_weight=(20e-3 / (10.0 * math.sqrt(num_neurons * connection_prob))),
        poisson_rate=50.0,
        poisson_weight=(
            (num_neurons * connection_prob * 20e-3)
            / (10.0 * math.sqrt(num_neurons * connection_prob))
        ),
        bin_rate=10e-3,
        min_delay=1e-3 * 2,
        max_delay=1e-3 * 4,
        refractory_period=1e-3 * 20,
    )


if __name__ == "__main__":
    simulation, device, num_neurons, connection_prob, repeat, num_threads = sys.argv[1:]
    num_neurons = int(num_neurons)
    connection_prob = float(connection_prob)

    torch.set_num_threads(int(num_threads))
    seed = get_deterministic_seed(int(repeat), num_neurons, connection_prob)

    timed_out = SIMULATIONS[simulation](
        build_sim_config(device, num_neurons, connection_prob), seed
    )
    if timed_out:
        sys.exit(TIMED_OUT_EXIT_CODE)
