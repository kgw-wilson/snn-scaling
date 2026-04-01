from dataclasses import dataclass
import math
import torch


@dataclass
class ERGraphConfig:
    """Configuration for Erdos-Renyi graphs"""

    num_neurons: int
    connection_prob: float
    global_coupling_strength: float
    device: torch.device
    dtype: torch.dtype

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
            not isinstance(self.global_coupling_strength, float)
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

        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"dtype must be a torch.dtype, got {type(self.dtype)}")


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

    timestep: float
    simulation_time: float
    membrane_time_constant: float
    synaptic_time_constant: float
    resting_voltage: float
    threshold_voltage: float
    poisson_rate: float
    bin_rate: float
    min_delay: float
    max_delay: float
    refractory_period: float

    @property
    def num_timesteps(self) -> int:
        return int(self.simulation_time / self.timestep)

    @property
    def membrane_decay(self):
        """
        Calculate membrane voltage decay factor based on timestep and membrane time constant

        Assumes every neuron is updated at each fixed timestep, which is not the case
        for event-driven simulations.
        """
        return math.exp(-self.timestep / self.membrane_time_constant)

    @property
    def synaptic_decay(self):
        """
        Calculate synaptic current decay factor based on timestep and synaptic time constant

        Assumes every neuron is updated at each fixed timestep, which is not the case
        for event-driven simulations.
        """
        return math.exp(-self.timestep / self.synaptic_time_constant)

    @property
    def membrane_bias(self):
        """
        Calculate resting bias term for more optimized voltage updates

        After some simple rearrangement, we see this biase term reduces
        an operation.

        V = (V - rest) * decay + rest + current
        V = V * decay - rest * decay + rest + current
        V = V * decay + rest - rest * decay + current
        V = V * decay + rest (1 - decay) + current
        V = V * decay + membrane_bias + current
        """
        return self.resting_voltage * (1 - self.membrane_decay)

    @property
    def poisson_prob(self) -> float:
        """Converts poisson spikes per second to poisson spikes per timestep"""
        return self.poisson_rate * self.timestep

    @property
    def num_bins(self) -> int:
        return math.ceil(self.simulation_time / self.bin_rate)

    @property
    def timesteps_per_bin(self) -> int:
        return int(self.num_timesteps / self.num_bins)

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

        if not isinstance(self.poisson_rate, float) or self.poisson_rate <= 0:
            raise TypeError(
                f"poisson_rate must be positive float, got {self.poisson_rate}"
            )

        if self.poisson_prob >= 1 or self.poisson_prob > 0.1:
            raise ValueError(
                "poisson_rate * timestep must be in (0, 0.1]."
                f"Got poisson_prob={self.poisson_prob:.4f}. "
                "Decrease timestep or poisson_rate."
            )

        if not isinstance(self.bin_rate, float) or self.bin_rate <= 0:
            raise ValueError(f"bin_rate must be a positive float, got {self.bin_rate}")

        if self.bin_rate > self.simulation_time:
            raise ValueError("bin_rate cannot be greater than total simulation time")

        if self.bin_rate < self.timestep:
            raise ValueError("bin_rate must be greater than timestep")

        if not isinstance(self.min_delay, float) or self.min_delay < 0:
            raise ValueError(
                f"min_delay must be a non-negative float, got {self.min_delay}"
            )

        if not isinstance(self.max_delay, float) or self.max_delay < 0:
            raise ValueError(
                f"max_delay must be a non-negative float, got {self.max_delay}"
            )

        if self.min_delay < self.timestep:
            print(
                "Warning: min_delay cannot be less than timestep in clock-driven simulations."
            )

        if self.max_delay < self.min_delay:
            raise ValueError("max_delay must be greater than or equal to min_delay.")

        if not isinstance(self.refractory_period, float) or self.refractory_period < 0:
            raise ValueError(
                f"refractory_period must be a non-negative float, got {self.refractory_period}"
            )

        if self.refractory_period < self.timestep:
            print(
                "Warning: refractory_period has no meaning in clock-driven simulations when less than timestep."
            )
