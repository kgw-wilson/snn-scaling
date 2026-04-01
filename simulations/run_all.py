import numpy as np
import torch

from shared.simulation_config import ERGraphConfig, SNNConfig
from simulations.clock_driven.dense_cpu import clock_driven_dense_cpu
from simulations.clock_driven.dense_gpu import clock_driven_dense_gpu
from simulations.clock_driven.sparse_cpu import clock_driven_sparse_cpu
# from simulations.event_driven.cpu import event_driven_cpu


_CONNECTION_PROBS = [0.1]
_NUM_NEURONS = [1000]

_BASE_SEED = 42

_DEVICE_TO_SIMULATIONS = {
    torch.device("cpu"): [
        clock_driven_dense_cpu,
        clock_driven_sparse_cpu,
    ],  # clock_driven_dense_cpu, clock_driven_sparse_cpu, event_driven_cpu
    torch.device("cuda"): [clock_driven_dense_gpu],
    torch.device("mps"): [],
}


def _get_available_devices() -> list[torch.device]:
    """Returns list of all available devices

    CPU is always available. Because CUDA index is not specified,
    assumes current CUDA device. MPS only surfaces one device.
    """

    available_devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        available_devices.append(torch.device("cuda"))

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
