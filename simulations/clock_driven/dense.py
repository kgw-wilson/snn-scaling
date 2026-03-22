import torch

from shared.graph_creation import create_er_dense
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_dense(graph_config: ERGraphConfig, snn_config: SNNConfig) -> None:
    """Run a clock-driven LIF spiking neural network simulation on dense graph"""

    weights = create_er_dense(config=graph_config)

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    timestep = snn_config.timestep
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    num_timesteps = snn_config.num_timesteps
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    poisson_prob = snn_config.poisson_prob
    bin_rate = snn_config.bin_rate
    num_bins = snn_config.num_bins

    membrane_voltages = torch.full(
        (num_neurons,), resting_voltage, device=device, dtype=dtype
    )
    synaptic_currents = torch.zeros(num_neurons, device=device, dtype=dtype)
    recurrent_spikes_bool = torch.zeros(num_neurons, device=device, dtype=torch.bool)
    poisson_spikes_bool = torch.rand((num_timesteps, num_neurons)) < poisson_prob

    spikes_per_neuron = torch.zeros(num_neurons, device=device, dtype=torch.int32)
    spikes_per_bin = torch.zeros(num_bins, device=device, dtype=torch.int32)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            membrane_voltages = (
                membrane_voltages * membrane_decay
                + synaptic_currents * (1 - membrane_decay)
            )

            combined_spikes_bool = torch.logical_or(
                poisson_spikes_bool[t], recurrent_spikes_bool
            )
            combined_spikes_float = combined_spikes_bool.to(dtype)

            synaptic_currents = (
                synaptic_currents * synaptic_decay + weights @ combined_spikes_float
            )

            recurrent_spikes_bool = membrane_voltages >= threshold_voltage
            membrane_voltages[recurrent_spikes_bool] = resting_voltage

            spikes_per_neuron += combined_spikes_bool
            current_time = t * timestep
            current_bin = int(current_time // bin_rate)
            spikes_per_bin[current_bin] += combined_spikes_bool.sum().item()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
