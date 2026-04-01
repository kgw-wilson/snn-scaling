import torch
from shared.simulation_config import ERGraphConfig, SNNConfig


def create_state_variables(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize dynamic neuron state variables for the LIF simulation

    Returns:
        membrane_voltages - membrane potential per neuron

        synaptic_currents - synaptic input per neuron

        last_spike_times - timestamp of last spike per neuron (for refractory enforcement)
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    resting_voltage = snn_config.resting_voltage

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=dtype
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=dtype)

    # Initialize to -inf to allow immediate firing
    last_spike_times = torch.full(
        (num_neurons,), -torch.inf, device=device, dtype=dtype
    )

    return membrane_voltages, synaptic_currents, last_spike_times
