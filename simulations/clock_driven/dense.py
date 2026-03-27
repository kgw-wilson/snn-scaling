import torch
from shared.utils import get_pytorch_compile_mode
from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_state_variables,
    allocate_spike_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_dense_clock_driven(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> None:
    """
    Run a clock-driven LIF spiking neural network simulation on dense graph

    Unpacking snn_config within the main loop causes a slight slowdown with
    attribute lookups but cleans up the code, so it is done. Random noise is
    generated here instead of in the compiled step function because
    .uniform_() can cause graph breaks and thus slowdowns.
    """

    bucketized_weights, bucket_offsets = build_dense_weights_bucketized_by_delay(
        graph_config, snn_config
    )

    ring_buffer, buffer_size = create_ring_buffer(graph_config, snn_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config, snn_config
    )

    random_noise, spikes_float = allocate_spike_tensors(graph_config=graph_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config, snn_config
    )

    compile_mode = get_pytorch_compile_mode(device=graph_config.device)
    compiled_step_func = torch.compile(_simulation_step, mode=compile_mode)

    with MonitoringWindow("simulation main loop"):

        for t in range(snn_config.num_timesteps):

            random_noise.uniform_()

            compiled_step_func(
                t=t,
                ring_buffer=ring_buffer,
                buffer_size=buffer_size,
                bucketized_weights=bucketized_weights,
                bucket_offsets=bucket_offsets,
                membrane_voltages=membrane_voltages,
                synaptic_currents=synaptic_currents,
                last_spike_times=last_spike_times,
                random_noise=random_noise,
                spikes_float=spikes_float,
                spikes_per_neuron=spikes_per_neuron,
                spikes_per_bin=spikes_per_bin,
                resting_voltage=snn_config.resting_voltage,
                membrane_bias=snn_config.membrane_bias,
                threshold_voltage=snn_config.threshold_voltage,
                membrane_decay=snn_config.membrane_decay,
                synaptic_decay=snn_config.synaptic_decay,
                poisson_prob=snn_config.poisson_prob,
                refractory_period=snn_config.refractory_period,
                bin_rate=snn_config.bin_rate,
                timestep=snn_config.timestep,
            )

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)


def _simulation_step(
    t: int,
    ring_buffer: torch.Tensor,
    buffer_size: int,
    bucketized_weights: torch.Tensor,
    bucket_offsets: torch.Tensor,
    membrane_voltages: torch.Tensor,
    synaptic_currents: torch.Tensor,
    last_spike_times: torch.Tensor,
    random_noise: torch.Tensor,
    spikes_float: torch.Tensor,
    spikes_per_neuron: torch.Tensor,
    spikes_per_bin: torch.Tensor,
    resting_voltage: float,
    membrane_bias: float,
    threshold_voltage: float,
    membrane_decay: float,
    synaptic_decay: float,
    poisson_prob: float,
    refractory_period: float,
    bin_rate: float,
    timestep: float,
) -> None:
    """
    Step forward the simulation by one timestep

    All operations need to be in-place as no tensors are returned by
    this function. Local variables don't seem to decrease performance
    compared to inlining so they are used for clarity.

    This is broken out of the main simulation loop in order to be
    compatible with PyTorch compilation. Compilation provides about
    a 20% speedup but the first call is slower because the generated
    machine code is reused by subsequent calls.
    """

    # time and indices
    current_time = t * timestep
    buffer_idx = t % buffer_size

    # in-place current & voltage updates
    synaptic_currents.mul_(synaptic_decay).add_(ring_buffer[buffer_idx])
    membrane_voltages.mul_(membrane_decay).add_(membrane_bias).add_(synaptic_currents)

    # spike generation
    poisson_spikes = random_noise < poisson_prob
    can_spike_mask = current_time - last_spike_times >= refractory_period
    recurrent_spikes = membrane_voltages >= threshold_voltage
    spikes_bool = (poisson_spikes | recurrent_spikes) & (can_spike_mask)
    spikes_float[:] = spikes_bool

    # propagation (schedule future current in buffer)
    bucket_indices_in_buffer = (buffer_idx + bucket_offsets) % buffer_size
    ring_buffer[bucket_indices_in_buffer] += bucketized_weights @ spikes_float

    # variable resets (.where seems to be slightly faster than masking with spikes_bool)
    ring_buffer[buffer_idx].zero_()
    membrane_voltages[:] = torch.where(spikes_bool, resting_voltage, membrane_voltages)
    last_spike_times[:] = torch.where(spikes_bool, current_time, last_spike_times)

    # reporting
    spikes_per_neuron += spikes_bool
    spikes_per_bin[int(current_time // bin_rate)] += spikes_bool.sum()
