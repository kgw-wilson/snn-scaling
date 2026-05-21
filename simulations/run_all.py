import math

from shared.simulation_config import SimulationConfig
from shared.utils import get_available_devices, get_deterministic_seed
from simulations.clock_driven_dense.runner import clock_driven_dense
from simulations.clock_driven_sparse.runner import clock_driven_sparse
from simulations.event_driven.runner import event_driven_cpu
from simulations.brian2.runner import clock_driven_brian2

_CONNECTION_PROBS = [0.01, 0.1, 0.3] #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
_NUM_NEURONS = [1000]
_NUM_REPEATS = 3
_BASE_SEED = 42

_DEVICE_STR_TO_SIMULATIONS = {
    "cpu": [
        clock_driven_dense,
        # clock_driven_sparse,
        # event_driven_cpu,
        clock_driven_brian2
    ],
    "gpu": [
        # clock_driven_dense,
        # clock_driven_sparse,
    ],
    "neuromorphic": [],
}


if __name__ == "__main__":

    available_device_strs = get_available_devices()

    for device_str in available_device_strs:

        for simulation in _DEVICE_STR_TO_SIMULATIONS[device_str]:

            for num_neurons in _NUM_NEURONS:

                for connection_prob in _CONNECTION_PROBS:

                    for repeat_i in range(_NUM_REPEATS):

                        print("===================")
                        print(
                            f"{simulation.__name__}, {device_str=}, {num_neurons=}, {connection_prob=}"
                        )

                        sim_config = SimulationConfig(
                            max_runtime=5 * 60,
                            num_neurons=num_neurons,
                            inhibitory_fraction=0.2,
                            connection_prob=connection_prob,
                            device_str=device_str,
                            timestep=0.1e-3,
                            simulation_time=1.0,
                            membrane_time_constant=10e-3,
                            resting_voltage=0.0,
                            threshold_voltage=10e-3,
                            weight=0.1e-3,
                            inhibitory_weight_multiplier=-5.0,
                            poisson_rate_multiplier=2.0,
                            bin_rate=10e-3,
                            min_delay=2e-3,
                            max_delay=4e-3,
                            refractory_period=2e-3,
                        )
                        seed = get_deterministic_seed(
                            repeat_i, sim_config.num_neurons, sim_config.connection_prob
                        )

                        simulation(sim_config, seed)

    print("All simulations completed.")
