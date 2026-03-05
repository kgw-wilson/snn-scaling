import math
import torch

from shared.graph_creation import create_er_dense
from shared.monitoring import MonitoringWindow
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_dense(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """
    Run a clock-driven LIF spiking neural network simulation on dense graph

    This implementation is designed for benchmarking CPU vs GPU performance
    for dense operations under controlled scaling conditions.

    All neurons are updated at every timestep using explicit Euler
    integration of the form dv/dt = (-v + I_syn) / tau_m. Computational costs are
    thus O(T * N^2) from dense matrix multiplication.

    Current-based LIF neurons with no conductances and no refractory periods.
    The model is minimal to isolate scaling behavior from biological complexity.

    Synaptic current decays exponentially and is incremented by recurrent
    spikes via weights @ spikes.

    Recording and weight updates are not done.
    """

    # Precompute constants to avoid work within the loop
    dtype = graph_config.dtype
    membrane_decay = snn_config.timestep / snn_config.membrane_time_constant
    synaptic_decay = math.exp(-snn_config.timestep / snn_config.synaptic_time_constant)

    weights = create_er_dense(graph_config)
    membrane_voltages = torch.full(
        (graph_config.num_neurons,),
        snn_config.resting_voltage,
        device=graph_config.device,
        dtype=dtype,
    )
    binary_spikes = torch.zeros(
        graph_config.num_neurons, device=graph_config.device, dtype=torch.bool
    )
    synaptic_currents = torch.zeros(
        graph_config.num_neurons, device=graph_config.device, dtype=dtype
    )

    with MonitoringWindow("simulation main loop"):

        for _ in range(snn_config.num_timesteps):

            membrane_voltages += (
                -membrane_voltages + synaptic_currents
            ) * membrane_decay
            binary_spikes = membrane_voltages >= snn_config.threshold_voltage
            membrane_voltages[binary_spikes] = snn_config.resting_voltage
            synaptic_currents = (
                synaptic_currents * synaptic_decay + weights @ binary_spikes.to(dtype)
            )


if __name__ == "__main__":

    graph_config = ERGraphConfig(
        seed=42,
        num_neurons=1000,
        connection_prob=0.1,
        global_coupling_strength=0.1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    snn_config = SNNConfig(
        timestep=0.1e-3,
        simulation_time=0.2,
        membrane_time_constant=20e-3,
        synaptic_time_constant=5e-3,
        resting_voltage=-70.0,
        threshold_voltage=-50.0,
    )

    torch.manual_seed(graph_config.seed)

    run_simulation_dense(graph_config, snn_config)
