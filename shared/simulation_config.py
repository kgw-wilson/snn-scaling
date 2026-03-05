from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ERGraphConfig:
    """Configuration for Erdos-Renyi graphs"""

    seed: int
    num_neurons: int
    connection_prob: float
    global_coupling_strength: float
    device: torch.device
    dtype: torch.dtype | np.dtype

    def __post_init__(self):
        """Validate values after instantiation"""

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")

        if not isinstance(self.num_neurons, int) or self.num_neurons <= 0:
            raise ValueError(
                f"number of neurons must be a positive integer, got {self.num_neurons}"
            )

        if (
            not isinstance(self.connection_prob, (float, int))
            or not 0 < self.connection_prob <= 1
        ):
            raise ValueError(
                f"connection probability must be in (0,1], got {self.connection_prob}"
            )

        if (
            not isinstance(self.global_coupling_strength, (float, int))
            or self.global_coupling_strength <= 0
        ):
            raise ValueError(
                f"global coupling trength must be positive, got {self.global_coupling_strength}"
            )

        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, got {type(self.device)}")
        else:
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available on this system")
                if (
                    self.device.index is not None
                    and self.device.index >= torch.cuda.device_count()
                ):
                    raise ValueError(f"CUDA device {self.device.index} not found")

            elif self.device.type == "mps":
                if not torch.backends.mps.is_available():
                    raise ValueError("MPS is not available on this system")
                if not torch.backends.mps.is_built():
                    raise ValueError("PyTorch was not built with MPS support")

        if not isinstance(self.dtype, (torch.dtype, np.dtype)):
            raise TypeError(f"dtype must be torch.dtype or np.dtype, got {self.dtype}")


@dataclass
class SNNConfig:
    """
    Spiking Neural Network configuration

    Time-related constants are in seconds. Voltage variables
    are in millivolts.
    """

    timestep: float
    simulation_time: float

    # How quickly membrane voltage decays toward resting voltage
    membrane_time_constant: float

    # How quickly synaptic current decays after a spike
    synaptic_time_constant: float

    resting_voltage: float
    threshold_voltage: float

    @property
    def num_timesteps(self) -> int:
        return int(self.simulation_time / self.timestep)

    def __post_init__(self):
        """Validate values after instantiation."""

        if not isinstance(self.timestep, (float, int)) or self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")

        if (
            not isinstance(self.simulation_time, (float, int))
            or self.simulation_time <= 0
        ):
            raise ValueError(
                f"simulation time must be positive, got {self.simulation_time}"
            )

        if self.simulation_time <= self.timestep:
            raise ValueError("simulation time must be greater than timestep")

        if (
            not isinstance(self.membrane_time_constant, (float, int))
            or self.membrane_time_constant <= 0
        ):
            raise ValueError(
                f"membrane time constant must be positive, got {self.membrane_time_constant}"
            )

        if (
            not isinstance(self.synaptic_time_constant, (float, int))
            or self.synaptic_time_constant <= 0
        ):
            raise ValueError(
                f"synaptic time constant must be positive, got {self.synaptic_time_constant}"
            )

        if self.timestep >= min(
            self.membrane_time_constant, self.synaptic_time_constant
        ):
            raise ValueError(
                "timestep should be smaller than both membrane time constant "
                "and synaptic time constant for numerical stability"
            )

        if not isinstance(self.resting_voltage, (float, int)):
            raise TypeError(
                f"resting voltage must be a number, got {type(self.resting_voltage)}"
            )

        if not isinstance(self.threshold_voltage, (float, int)):
            raise TypeError(
                f"threshold voltage must be a number, got {type(self.threshold_voltage)}"
            )

        if self.threshold_voltage <= self.resting_voltage:
            raise ValueError("threshold voltage must be greater than resting voltage")
