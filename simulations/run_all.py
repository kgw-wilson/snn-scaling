import numpy as np
import torch

from shared.simulation_config import ERGraphConfig, SNNConfig
from simulations.clock_driven.dense import run_simulation_dense
from simulations.clock_driven.sparse_cpu import run_simulation_sparse_cpu
from simulations.clock_driven.sparse_gpu import run_simulation_sparse_gpu


_CONNECTION_PROBS = [0.1]
_NUM_NEURONS = [1000]

_BASE_SEED = 42

_DEVICE_TO_SIMULATIONS = {
    torch.device("cpu"): [run_simulation_dense, run_simulation_sparse_cpu],
    torch.device("cuda"): [run_simulation_dense, run_simulation_sparse_gpu],
    torch.device("mps"): [run_simulation_dense],
}


def _get_available_devices() -> list[torch.device]:
    """Returns list of all available devices

    CPU is always available. Because CUDA index is not specified,
    assumes current CUDA device. MPS only surfaces one device.
    """

    available_devices = [torch.device("cpu")]

    # if torch.cuda.is_available():
    # available_devices.append(torch.device("cuda"))

    # if torch.backends.mps.is_available():
    #     available_devices.append(torch.device("mps"))

    return available_devices


if __name__ == "__main__":

    available_devices = _get_available_devices()

    for device in available_devices:

        for simulation in _DEVICE_TO_SIMULATIONS[device]:

            for num_neurons in _NUM_NEURONS:

                for connection_prob in _CONNECTION_PROBS:

                    print("===================")
                    print(
                        f"Running {simulation.__name__} on {device} with {num_neurons=} and {connection_prob=}"
                    )

                    # Ensure same configuration always produces same seed
                    seed = (
                        hash(
                            (
                                _BASE_SEED,
                                simulation.__name__,
                                num_neurons,
                                connection_prob,
                            )
                        )
                        % 2**32
                    )
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    graph_config = ERGraphConfig(
                        num_neurons=num_neurons,
                        connection_prob=connection_prob,
                        global_coupling_strength=10.0,
                        device=device,
                        dtype=torch.float32,
                    )

                    snn_config = SNNConfig(
                        timestep=0.1e-3,
                        simulation_time=1.0,
                        membrane_time_constant=20e-3,
                        synaptic_time_constant=5e-3,
                        resting_voltage=0.0,
                        threshold_voltage=20.0,
                        poisson_rate=5.0,
                        bin_rate=1e-3,
                        min_delay=0.1e-3,  # 2.1e-3,
                        max_delay=0.4e-3,  # 4.3e-3,
                        refractory_period=0.2e-3,
                    )

                    simulation(graph_config=graph_config, snn_config=snn_config)

    print("All simulations completed.")
