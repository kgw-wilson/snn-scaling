import torch
from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import SimulationConfig
from shared.utils import create_state_variables


def clock_driven_dense_gpu_compiled(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation on GPU using dense graph for weights"""

    torch.manual_seed(seed)

    timestep = sim_config.timestep
    num_timesteps = sim_config.num_timesteps
    timesteps_per_bin = sim_config.timesteps_per_bin
    resistance = sim_config.resistance
    resting_voltage = sim_config.resting_voltage
    threshold_voltage = sim_config.threshold_voltage
    membrane_decay = sim_config.membrane_decay
    synaptic_decay = sim_config.synaptic_decay
    poisson_weight = sim_config.poisson_weight
    poisson_prob = sim_config.poisson_prob
    refractory_period = sim_config.refractory_period

    bucketized_weights = build_dense_weights_bucketized_by_delay(sim_config)

    ring_buffer, buffer_size = create_ring_buffer(sim_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        sim_config
    )

    random_noise, _ = create_spike_tensors(sim_config)

    (
        _,
        _,
        bucket_indices_in_buffer,
    ) = create_lookup_tensors(sim_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    one_minus_decay = 1.0 - membrane_decay

    buffer_slice = torch.zeros_like(ring_buffer[0])


    simulate_fn = torch.compile(
        _simulate,
        backend="inductor",
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    with MonitoringWindow("Simulation main"):

        simulate_fn(
            num_timesteps,
            timestep,
            buffer_size,
            synaptic_currents,
            random_noise,
            synaptic_decay,
            ring_buffer,
            buffer_slice,
            poisson_weight,
            poisson_prob,
            last_spike_times,
            refractory_period,
            resistance,
            resting_voltage,
            membrane_decay,
            one_minus_decay,
            membrane_voltages,
            threshold_voltage,
            bucket_indices_in_buffer,
            bucketized_weights,
            spikes_per_neuron,
            spikes_per_bin,
            timesteps_per_bin,
        )

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)


def _simulate(
    num_timesteps: int,
    timestep: float,
    buffer_size: int,
    synaptic_currents: torch.Tensor,
    random_noise: torch.Tensor,
    synaptic_decay: float,
    ring_buffer: torch.Tensor,
    buffer_slice: torch.Tensor,
    poisson_weight: float,
    poisson_prob: float,
    last_spike_times: torch.Tensor,
    refractory_period: float,
    resistance: float,
    resting_voltage: float,
    membrane_decay: float,
    one_minus_decay: float,
    membrane_voltages: torch.Tensor,
    threshold_voltage: float,
    bucket_indices_in_buffer: torch.Tensor,
    bucketized_weights: torch.Tensor,
    spikes_per_neuron: torch.Tensor,
    spikes_per_bin: torch.Tensor,
    timesteps_per_bin: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    buffer_index = 0

    for t in range(num_timesteps):

        random_noise = torch.rand_like(random_noise)

        current_time = t * timestep

        buffer_index = (buffer_index + 1) % buffer_size

        synaptic_currents = (
            synaptic_currents * synaptic_decay
            + ring_buffer[buffer_index]
            + poisson_weight * (random_noise < poisson_prob)
        )

        outside_refractory = (current_time - last_spike_times) >= refractory_period

        alpha = synaptic_currents * resistance + resting_voltage
        new_voltages = alpha * one_minus_decay + membrane_voltages * membrane_decay

        membrane_voltages = torch.where(
            outside_refractory,
            new_voltages,
            membrane_voltages,
        )

        spikes_bool = membrane_voltages >= threshold_voltage

        ring_buffer.index_add_(
            0,
            bucket_indices_in_buffer[t],
            bucketized_weights @ spikes_bool.to(torch.float32),
        )

        buffer_slice.copy_(ring_buffer[buffer_index])

        ring_buffer.index_add_(
            0,
            buffer_index,
            -buffer_slice,
        )

        membrane_voltages = torch.where(
            spikes_bool,
            resting_voltage,
            membrane_voltages,
        )

        last_spike_times = torch.where(
            spikes_bool,
            current_time,
            last_spike_times,
        )

        spikes_per_neuron += spikes_bool
        spikes_per_bin[t // timesteps_per_bin] += spikes_bool.sum()
