import torch
from shared.simulation_config import SimulationConfig


def create_state_variables(
    sim_config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize dynamic neuron state variables for the LIF simulation

    Returns:
        membrane_voltages - membrane potential per neuron

        synaptic_currents - synaptic input per neuron

        last_spike_times - timestamp of last spike per neuron (for refractory enforcement)
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    dtype = sim_config.dtype
    resting_voltage = sim_config.resting_voltage

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=dtype
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=dtype)

    # Initialize to -inf to allow immediate firing
    last_spike_times = torch.full(
        (num_neurons,), -torch.inf, device=device, dtype=dtype
    )

    return membrane_voltages, synaptic_currents, last_spike_times
