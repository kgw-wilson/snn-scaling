import hashlib

import torch

from shared.simulation_config import SimulationConfig


def create_state_variables(
    sim_config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize dynamic neuron state variables for an LIF simulation

    Returns:
        membrane_voltages - membrane potential per neuron

        synaptic_currents - synaptic input per neuron

        last_spike_times - timestamp of last spike per neuron (for refractory enforcement)
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    resting_voltage = sim_config.resting_voltage

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=torch.float32
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=torch.float32)

    # Initialize to -inf to allow immediate firing
    last_spike_times = torch.full(
        (num_neurons,), -torch.inf, device=device, dtype=torch.float32
    )

    return membrane_voltages, synaptic_currents, last_spike_times


def create_delays(sim_config: SimulationConfig) -> torch.Tensor:
    """Return tensor with uniform delay values for each neuron pair"""

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    min_delay = sim_config.min_delay
    max_delay = sim_config.max_delay

    delays = torch.empty(num_neurons, num_neurons, device=device, dtype=torch.float32)
    delays.uniform_(min_delay, max_delay)

    return delays


def get_available_devices() -> list[str]:
    """Returns list of all available device names"""

    available_devices = ["cpu"]

    if torch.cuda.is_available():
        available_devices.append("gpu")

    if _has_spinnaker_board():
        available_devices.append("neuromorphic")

    return available_devices


def get_deterministic_seed(*args) -> int:
    """Generates seed that's constant between interpreter runs for the given args"""

    data = repr(args).encode()
    hashed_bytes = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(hashed_bytes) % (2**32)


def _has_spinnaker_board() -> bool:
    """True only if sPyNNaker can reach real hardware (not a virtual board)

    This check is hacky, but the package does not appear to surface an official
    means for checking access to hardware. Any failure means no usable board.
    """

    try:
        import pyNN.spiNNaker as p

        p.setup()
        try:
            return "virtual" not in str(p.get_machine()).lower()
        finally:
            p.end()
    except Exception:
        return False
