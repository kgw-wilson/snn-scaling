import time
import psutil
from dataclasses import dataclass
import math
import torch

from shared.graph_creation import create_er_dense
from shared.simulation_config import ERGraphConfig, SNNConfig


@dataclass
class MeasurementResult:
    """Standardized measurement results"""

    runtime: float  # seconds
    peak_memory: float  # bytes
    energy: float  # joules


class CPUMeasurements:
    """CPU timing, memory, and energy measurements using pyRAPL and psutil"""

    def __init__(self):
        import pyRAPL

        pyRAPL.setup()
        self.pyrapl_meter = pyRAPL.Measurement("cpu_energy")
        self.process = psutil.Process()
        self.start_time = None
        self.mem_start = None

    def begin(self):
        """Start CPU measurements"""
        self.mem_start = self.process.memory_info().rss
        self.start_time = time.perf_counter()
        self.pyrapl_meter.begin()

    def end(self) -> MeasurementResult:
        """Stop measurements and return standardized results"""
        self.pyrapl_meter.end()
        runtime = time.perf_counter() - self.start_time
        peak_memory = self.process.memory_info().rss - self.mem_start
        energy = self.pyrapl_meter.result.pkg  # joules
        return MeasurementResult(
            runtime=runtime, peak_memory=peak_memory, energy=energy
        )


class GPUMeasurements:
    """GPU timing, memory, and energy measurements using torch.cuda and pynvml"""

    def __init__(self, device: torch.device):
        if device.type != "cuda":
            raise ValueError("GPUMeasurements requires a CUDA device")
        import pynvml

        pynvml.nvmlInit()
        self.pynvml = pynvml
        self.device = device
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        self.start_event = None
        self.end_event = None
        self.power_start_mW = None

    def begin(self):
        """Start GPU measurements"""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        self.power_start_mW = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)

    def end(self) -> MeasurementResult:
        """Stop GPU measurements and return standardized results"""
        torch.cuda.synchronize()
        self.end_event.record()
        torch.cuda.synchronize()
        runtime = self.start_event.elapsed_time(self.end_event) / 1000.0  # ms -> s
        peak_memory = torch.cuda.max_memory_allocated(self.device)  # bytes
        power_end_mW = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
        energy = (power_end_mW / 1000.0) * runtime  # rough estimate: W * s = J
        return MeasurementResult(
            runtime=runtime, peak_memory=peak_memory, energy=energy
        )


def begin_measurements(device: torch.device):
    """
    Factory function to start measurements for a given device.
    Returns an instance of CPUMeasurements or GPUMeasurements.
    """
    if device.type == "cpu":
        meas = CPUMeasurements()
    elif device.type == "cuda":
        meas = GPUMeasurements(device)
    else:
        raise ValueError(f"Unsupported device type: {device.type}")
    meas.begin()
    return meas


def end_measurements(measurement_obj) -> None:
    """Stop the measurements and print standardized results"""
    print(measurement_obj.end())


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

    Recording and weight updates are not done, since the function is purely
    for runtime benchmarking.

    Measures runtime, peak memory usage, and approximate energy consumed for
    the main simulation loop.
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

    begin_measurements(graph_config.device)

    for _ in range(snn_config.num_timesteps):

        membrane_voltages += (-membrane_voltages + synaptic_currents) * membrane_decay

        binary_spikes = membrane_voltages >= snn_config.threshold_voltage

        membrane_voltages[binary_spikes] = snn_config.resting_voltage

        synaptic_currents = (
            synaptic_currents * synaptic_decay + weights @ binary_spikes.to(dtype)
        )

    end_measurements(graph_config.device)


if __name__ == "__main__":

    graph_config = ERGraphConfig(
        seed=42,
        num_neurons=10000,
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

    run_simulation_dense(graph_config, snn_config)
