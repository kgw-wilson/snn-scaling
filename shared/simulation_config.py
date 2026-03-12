from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class ERGraphConfig:
    """Configuration for Erdos-Renyi graphs"""

    num_neurons: int
    connection_prob: float
    global_coupling_strength: float
    device: torch.device

    def __post_init__(self):
        """Validate values after instantiation"""

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


@dataclass
class SNNConfig:
    """
    Spiking Neural Network configuration

    Time-related constants are in seconds. Voltage variables
    are in millivolts. Rates are in Hz.

    Membrane/synatic time constants determine how quickly membrane
    voltage or synaptic current, respectively, decays after a spike.
    Poisson variables determine the frequency and strength of external
    Poisson input which drives activity in the network. The Bernoulli
    appoximation to a Poisson process is used, so poisson_prob
    should be << 1 for accuracy.
    """

    timestep: float | None
    simulation_time: float
    membrane_time_constant: float
    synaptic_time_constant: float
    resting_voltage: float
    threshold_voltage: float
    poisson_rate: float
    poisson_weight: float

    @property
    def num_timesteps(self) -> int:
        return int(self.simulation_time / self.timestep)

    @property
    def membrane_decay(self):
        """
        Calculate membrane voltage decay factor based on timestep and membrane time constant

        Assumes a fixed timestep, which is not the case for event-driven simulations.
        """
        return math.exp(-self.timestep / self.membrane_time_constant)

    @property
    def synaptic_decay(self):
        """
        Calculate synaptic current decay factor based on timestep and synaptic time constant

        Assumes a fixed timestep, which is not the case for event-driven simulations.
        """
        return math.exp(-self.timestep / self.synaptic_time_constant)

    @property
    def poisson_prob(self) -> float:
        return self.poisson_rate * self.timestep

    def __post_init__(self):
        """Validate values after instantiation."""

        if not isinstance(self.timestep, float) or self.timestep <= 0:
            raise ValueError(f"timestep must be positive float, got {self.timestep}")

        if not isinstance(self.simulation_time, float) or self.simulation_time <= 0:
            raise ValueError(
                f"simulation time must be positive float, got {self.simulation_time}"
            )

        if self.simulation_time <= self.timestep:
            raise ValueError("simulation time must be greater than timestep")

        if (
            not isinstance(self.membrane_time_constant, float)
            or self.membrane_time_constant <= 0
        ):
            raise ValueError(
                f"membrane time constant must be positive float, got {self.membrane_time_constant}"
            )

        if (
            not isinstance(self.synaptic_time_constant, float)
            or self.synaptic_time_constant <= 0
        ):
            raise ValueError(
                f"synaptic time constant must be positive float, got {self.synaptic_time_constant}"
            )

        if self.timestep >= min(
            self.membrane_time_constant, self.synaptic_time_constant
        ):
            raise ValueError(
                "timestep should be smaller than both membrane time constant "
                "and synaptic time constant for numerical stability"
            )

        if not isinstance(self.resting_voltage, float):
            raise TypeError(
                f"resting voltage must be float, got {type(self.resting_voltage)}"
            )

        if not isinstance(self.threshold_voltage, float):
            raise TypeError(
                f"threshold voltage must be float, got {type(self.threshold_voltage)}"
            )

        if self.threshold_voltage <= self.resting_voltage:
            raise ValueError("threshold voltage must be greater than resting voltage")

        if not isinstance(self.poisson_rate, float):
            raise TypeError(
                f"poisson_rate must be float, got {type(self.poisson_rate)}"
            )

        if self.poisson_rate <= 0:
            raise ValueError(f"poisson_rate must be positive, got {self.poisson_rate}")

        if not isinstance(self.poisson_weight, float):
            raise TypeError(
                f"poisson_weight must be float, got {type(self.poisson_weight)}"
            )

        if self.poisson_weight <= 0:
            raise ValueError(
                f"poisson_weight must be positive, got {self.poisson_weight}"
            )

        if self.poisson_prob >= 1 or self.poisson_prob > 0.1:
            raise ValueError(
                "poisson_rate * timestep must be in (0, 0.1]."
                f"Got poisson_prob={self.poisson_prob:.4f}. "
                "Decrease timestep or poisson_rate."
            )
