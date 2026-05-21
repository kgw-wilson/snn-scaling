from dataclasses import dataclass
import math
import torch


@dataclass
class SimulationConfig:
    """
    Simulation configuration parameters for spiking neural network simulations

    Time-related constants are in seconds. Voltage variables
    are in millivolts. Rates are in Hz.
    """

    max_runtime: int
    num_neurons: int
    inhibitory_fraction: float
    connection_prob: float
    device_str: str
    timestep: float
    simulation_time: float
    membrane_time_constant: float
    inhibitory_weight_multiplier: float
    resting_voltage: float
    threshold_voltage: float
    poisson_rate_multiplier: float
    weight: float
    bin_rate: float
    min_delay: float
    max_delay: float
    refractory_period: float

    @property
    def device(self) -> torch.device | None:
        if self.device_str == "cpu":
            return torch.device("cpu")
        if self.device_str == "gpu":
            return torch.device("cuda")
        return None

    @property
    def num_inhibitory(self) -> int:
        return int(self.num_neurons * self.inhibitory_fraction)

    @property
    def num_excitatory(self) -> int:
        return self.num_neurons - self.num_inhibitory

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
    def poisson_rate(self) -> float:
        """
        Calculate poisson input rate

        Uses https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_2000.html
        """
        num_excitatory_conn = self.connection_prob * self.num_excitatory
        threshold_rate = self.threshold_voltage / (
            self.weight * num_excitatory_conn * self.membrane_time_constant
        )
        return self.poisson_rate_multiplier * threshold_rate

    @property
    def poisson_prob(self) -> float:
        """Converts poisson spikes per second to probability of poisson spike per timestep"""
        # return self.poisson_rate * self.timestep
        return self.poisson_rate * self.timestep * (self.connection_prob * self.num_excitatory)

    @property
    def num_bins(self) -> int:
        return math.ceil(self.simulation_time / self.bin_rate)

    @property
    def timesteps_per_bin(self) -> int:
        return math.ceil(self.num_timesteps / self.num_bins)

    def __post_init__(self):
        """Validate values after instantiation."""

        if not isinstance(self.max_runtime, int):
            raise ValueError(
                f"max_runtime must be a positive int, got {self.max_runtime}"
            )

        if not isinstance(self.num_neurons, int) or self.num_neurons <= 0:
            raise ValueError(
                f"number of neurons must be a positive integer, got {self.num_neurons}"
            )

        if not isinstance(self.connection_prob, float) or not self.connection_prob > 0:
            raise ValueError(
                f"connection probability must be float > 0, got {self.connection_prob}"
            )

        if (
            not isinstance(self.inhibitory_weight_multiplier, float)
            or self.inhibitory_weight_multiplier > 0
        ):
            raise ValueError(
                f"inhibitory weight multiplier must be non-positive float, got {self.inhibitory_weight_multiplier}"
            )

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

        if self.timestep >= self.membrane_time_constant:
            raise ValueError(
                "timestep should be smaller than membrane time constant for numerical stability"
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

        if not isinstance(self.weight, float) or self.weight <= 0:
            raise TypeError(f"weight must be positive float, got {self.weight}")

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
