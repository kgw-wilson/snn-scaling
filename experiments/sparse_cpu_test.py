import numpy as np
import math
import torch

from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.graph_creation import create_er_sparse_cpu
from shared.monitoring import MonitoringWindow


def run_simulation_sparse_cpu(graph_config, snn_config):
    """Run SNN simulation using sparse csr matrix for synaptic weights on CPU"""

    weights = create_er_sparse_cpu(graph_config)

    membrane_decay = snn_config.timestep / snn_config.membrane_time_constant
    synaptic_decay = math.exp(-snn_config.timestep / snn_config.synaptic_time_constant)

    membrane_voltages = np.full(
        graph_config.num_neurons, snn_config.resting_voltage, dtype=graph_config.dtype
    )
    synaptic_currents = np.zeros(graph_config.num_neurons, dtype=graph_config.dtype)
    binary_spikes = np.zeros(graph_config.num_neurons, dtype=bool)

    with MonitoringWindow("simulation main loop"):

        for _ in range(snn_config.num_timesteps):
            membrane_voltages += (
                -membrane_voltages + synaptic_currents
            ) * membrane_decay
            binary_spikes = membrane_voltages >= snn_config.threshold_voltage
            membrane_voltages[binary_spikes] = snn_config.resting_voltage

            spike_vector = binary_spikes.astype(graph_config.dtype).reshape(-1, 1)
            synaptic_input = weights.dot(spike_vector).reshape(-1)
            synaptic_currents = synaptic_currents * synaptic_decay + synaptic_input


if __name__ == "__main__":

    graph_config = ERGraphConfig(
        seed=42,
        num_neurons=1000,
        connection_prob=0.1,
        global_coupling_strength=0.1,
        device=torch.device("cpu"),
        dtype=np.dtype(np.float32),
    )

    snn_config = SNNConfig(
        timestep=0.1e-3,
        simulation_time=0.2,
        membrane_time_constant=20e-3,
        synaptic_time_constant=5e-3,
        resting_voltage=-70.0,
        threshold_voltage=-50.0,
    )

    np.random.seed(graph_config.seed)

    run_simulation_sparse_cpu(graph_config, snn_config)
