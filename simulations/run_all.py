import math

import torch

from shared.reporting import RESULTS_PATH
from shared.simulation_config import SimulationConfig
from shared.utils import get_available_devices, get_deterministic_seed
from simulations.clock_driven_dense.runner import clock_driven_dense
from simulations.clock_driven_openmp.runner import clock_driven_openmp
from simulations.clock_driven_sparse.runner import clock_driven_sparse
from simulations.event_driven.runner import event_driven_cpu

_CONNECTION_PROBS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
_NUM_NEURONS = [10, 100, 1000, 10000]
_NUM_REPEATS = 3
_NUM_THREADS = 22  # physical cores on the Xeon E5-2696 v4

_DEVICE_TO_SIMULATIONS = {
    "cpu": [
        clock_driven_dense,
        clock_driven_sparse,
        clock_driven_openmp,
        event_driven_cpu,
    ],
    "gpu": [
        clock_driven_dense,
        clock_driven_sparse,
    ],
    "neuromorphic": [],
}


if __name__ == "__main__":

    if RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH} exists, move it before starting a new run."
        )

    torch.set_num_threads(_NUM_THREADS)

    available_devices = get_available_devices()

    for device in available_devices:

        for simulation in _DEVICE_TO_SIMULATIONS[device]:

            for num_neurons in _NUM_NEURONS:

                for connection_prob in _CONNECTION_PROBS:

                    for repeat_i in range(_NUM_REPEATS):

                        print("===================")
                        print(
                            f"{simulation.__name__}, {device=}, {num_neurons=}, {connection_prob=}"
                        )

                        seed = get_deterministic_seed(
                            repeat_i, num_neurons, connection_prob
                        )

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
                        sim_config = SimulationConfig(
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
                            recurrent_weight=(
                                20e-3
                                / (10.0 * math.sqrt(num_neurons * connection_prob))
                            ),
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

                        simulation(sim_config, seed)

    print("All simulations completed.")
