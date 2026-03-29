import torch
from shared.utils import get_pytorch_compile_mode
from shared.clock_driven import (
    build_dense_weights_bucketized_by_delay,
    create_ring_buffer,
    create_state_variables,
    create_spike_tensors,
    create_lookup_tensors,
)
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig


def clock_driven_dense_gpu(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """Run a clock-driven LIF spiking neural network simulation on dense graph"""

    # Unpack config to avoid attribute lookups in simulation loop
    num_timesteps = snn_config.num_timesteps
    resting_voltage = snn_config.resting_voltage
    membrane_bias = snn_config.membrane_bias
    threshold_voltage = snn_config.threshold_voltage
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    poisson_prob = snn_config.poisson_prob
    refractory_period = snn_config.refractory_period

    bucketized_weights, bucket_offsets = build_dense_weights_bucketized_by_delay(
        graph_config, snn_config
    )

    ring_buffer = create_ring_buffer(graph_config, snn_config)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config, snn_config
    )

    random_noise, _ = create_spike_tensors(graph_config=graph_config)

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config, snn_config
    )

    timestep_values, bin_indices = create_lookup_tensors(graph_config, snn_config)

    compile_mode = get_pytorch_compile_mode(device=graph_config.device)
    compiled_step_func = torch.compile(_simulation_step, mode=compile_mode)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            random_noise.uniform_()

            synaptic_currents, membrane_voltages, weighted_spikes, spikes_bool = (
                compiled_step_func(
                    current_time=timestep_values[t],
                    ring_buffer=ring_buffer,
                    bucketized_weights=bucketized_weights,
                    membrane_voltages=membrane_voltages,
                    synaptic_currents=synaptic_currents,
                    last_spike_times=last_spike_times,
                    random_noise=random_noise,
                    synaptic_decay=synaptic_decay,
                    membrane_decay=membrane_decay,
                    membrane_bias=membrane_bias,
                    poisson_prob=poisson_prob,
                    refractory_period=refractory_period,
                    threshold_voltage=threshold_voltage,
                    resting_voltage=resting_voltage,
                )
            )
            ring_buffer[bucket_offsets] = weighted_spikes
            ring_buffer[:-1] = ring_buffer[1:].clone()
            ring_buffer[-1].zero_()
            spikes_per_neuron += spikes_bool
            spikes_per_bin[bin_indices[t]] += spikes_bool.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)


def _simulation_step(
    current_time: float,
    ring_buffer: torch.Tensor,
    bucketized_weights: torch.Tensor,
    membrane_voltages: torch.Tensor,
    synaptic_currents: torch.Tensor,
    last_spike_times: torch.Tensor,
    random_noise: torch.Tensor,
    synaptic_decay: float,
    membrane_decay: float,
    membrane_bias: float,
    poisson_prob: float,
    refractory_period: float,
    threshold_voltage: float,
    resting_voltage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Step forward the simulation by one timestep

    Operations should not be in-place because cudagraphs need input to
    not be mutated. For this same reason, dynamic indexing to update the
    ring buffer and reporting tesnors is done outside of this function.

    This is broken out of the main simulation loop in order to be
    compatible with PyTorch compilation. Compilation provides about
    a 20% speedup on CPU but the first call is slower because the
    generated machine code is reused by subsequent calls.

    Returns:
        synaptic_currents - updated state variable storing incoming current for
            all neurons at the next timestep

        membrane_voltages - updated state variable storing membrane voltages for
            all neurons at the next timestep

        weighted_spikes - result of matrix multiplication between
            bucketized_weights and spikes_bool

        spikes_bool - boolean Tensor storing spikes generated at this timestep
    """

    synaptic_currents = synaptic_currents * synaptic_decay + ring_buffer[0]
    membrane_voltages = (
        membrane_voltages * membrane_decay + membrane_bias + synaptic_currents
    )

    poisson_spikes = random_noise < poisson_prob
    can_spike_mask = current_time - last_spike_times >= refractory_period
    recurrent_spikes = membrane_voltages >= threshold_voltage
    spikes_bool = (poisson_spikes | recurrent_spikes) & (can_spike_mask)

    weighted_spikes = bucketized_weights @ spikes_bool

    membrane_voltages[spikes_bool] = resting_voltage
    last_spike_times[spikes_bool] = current_time

    return synaptic_currents, membrane_voltages, weighted_spikes, spikes_bool
