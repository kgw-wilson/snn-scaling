import torch

from shared.graph_creation import create_er_sparse
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics
from shared.simulation_config import ERGraphConfig, SNNConfig

def run_simulation_event_driven(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run an event-driven LIF spiking neural network simulation"""

    weights = create_er_sparse(config=graph_config, use_numpy=False)

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    dtype = graph_config.dtype
    num_timesteps = snn_config.num_timesteps
    timestep = snn_config.timestep
    membrane_time_constant = snn_config.membrane_time_constant
    synaptic_time_constant = snn_config.synaptic_time_constant
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    poisson_prob = snn_config.poisson_prob
    bin_rate = snn_config.bin_rate
    num_bins = snn_config.num_bins

    poisson_spikes_bool = torch.rand((num_timesteps, num_neurons)) < poisson_prob
    recurrent_spikes_bool = torch.zeros((num_neurons), dtype=torch.bool)
    last_update_times = torch.zeros(num_neurons, dtype=dtype)
    input_buffer = torch.zeros(num_neurons, dtype=dtype)

    crow = weights.crow_indices()
    cols = weights.col_indices()
    vals = weights.values()

    membrane_voltages = torch.full((num_neurons,), resting_voltage, dtype=dtype)
    synaptic_currents = torch.zeros(num_neurons, dtype=dtype)

    spikes_per_neuron = torch.zeros(num_neurons, dtype=torch.int32)
    spikes_per_bin = torch.zeros(num_bins, dtype=torch.int32)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            current_time = t * timestep

            # For each timestep, get poisson spikes
            current_poisson_spikes = poisson_spikes_bool[t]

            # Combine into one spikes array (only one spike per-neuron per-timestep)
            spike_emissions = torch.logical_or(current_poisson_spikes, recurrent_spikes_bool)
            spiking_indices = torch.where(spike_emissions)[0]

            # zero out membrane voltages for spiking neurons before updating connections
            # which could include currrently spiking neurons
            membrane_voltages[spiking_indices] = resting_voltage

            spikes_per_neuron += spike_emissions
            current_bin = int(current_time // bin_rate)
            spikes_per_bin[current_bin] += spike_emissions

            # For all spiking neurons at this timestep, update post-synaptic neurons
            for spiking_neuron_idx in spiking_indices:

                connections_start = crow[spiking_neuron_idx].item()
                connections_end = crow[spiking_neuron_idx + 1].item()
                post_synaptic_indices = cols[connections_start:connections_end]
                post_synaptic_weights = vals[connections_start:connections_end]
                input_buffer[post_synaptic_indices] += post_synaptic_weights
            
            active_neuron_mask = input_buffer != 0

            times_since_last_update = current_time - last_update_times[active_neuron_mask]
            synaptic_decays = torch.exp(-times_since_last_update / synaptic_time_constant)
            membrane_decays = torch.exp(-times_since_last_update / membrane_time_constant)

            updated_currents = synaptic_currents[active_neuron_mask] * synaptic_decays + input_buffer[active_neuron_mask]
            updated_voltages = membrane_voltages[
                active_neuron_mask
            ] * membrane_decays + updated_currents * (1 - membrane_decays)

            synaptic_currents[active_neuron_mask] = updated_currents
            membrane_voltages[active_neuron_mask] = updated_voltages
            last_update_times[active_neuron_mask] = current_time

            recurrent_spikes_bool = membrane_voltages >= threshold_voltage
            input_buffer.zero_()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
