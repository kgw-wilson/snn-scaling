import torch

from shared.graph_creation import create_er_sparse
from shared.event_driven import EventQueue
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

    event_queue = EventQueue()
    poisson_spikes_bool = torch.rand((num_timesteps, num_neurons)) < poisson_prob
    spike_indices = torch.nonzero(poisson_spikes_bool)
    crow = weights.crow_indices()
    cols = weights.col_indices()
    vals = weights.values()
    events = []
    for t, neuron_idx in spike_indices.tolist():

        current_time = t * timestep + delay
        connections_start = crow[neuron_idx]  # .item()
        connections_end = crow[neuron_idx + 1]  # .item()
        post_synaptic_indices = cols[connections_start:connections_end]
        post_synaptic_weights = vals[connections_start:connections_end]

        for target_idx, weight in zip(post_synaptic_indices, post_synaptic_weights):
            events.append((current_time, int(target_idx), float(weight)))

    event_queue.initialize_heap(events=events)

    membrane_voltages = torch.full((num_neurons,), resting_voltage, dtype=dtype)
    synaptic_currents = torch.zeros(num_neurons, dtype=dtype)
    last_update_times = torch.zeros(num_neurons, dtype=dtype)

    spikes_per_neuron = torch.zeros(num_neurons, dtype=torch.int32)
    spikes_per_bin = torch.zeros(num_bins, dtype=torch.int32)

    with MonitoringWindow("simulation main loop"):

        while len(event_queue) > 0:

            current_time, neuron_idx, weight = event_queue.pop()

            time_since_last_update = current_time - last_update_times[neuron_idx]
            synaptic_decay = torch.exp(-time_since_last_update / synaptic_time_constant)
            membrane_decay = torch.exp(-time_since_last_update / membrane_time_constant)

            updated_current = synaptic_currents[neuron_idx] * synaptic_decay + weight
            updated_voltage = membrane_voltages[
                neuron_idx
            ] * membrane_decay + updated_current * (1 - membrane_decay)

            synaptic_currents[neuron_idx] = updated_current
            membrane_voltages[neuron_idx] = updated_voltage
            last_update_times[neuron_idx] = current_time

            if updated_voltage >= threshold_voltage:

                arrival_time = current_time + delay
                connections_start = crow[neuron_idx].item()
                connections_end = crow[neuron_idx + 1].item()
                post_synaptic_indices = cols[connections_start:connections_end]
                post_synaptic_weights = vals[connections_start:connections_end]

                for target_idx, connection_weight in zip(
                    post_synaptic_indices, post_synaptic_weights
                ):
                    event_queue.push(
                        (arrival_time, target_idx.item(), connection_weight.item())
                    )

                membrane_voltages[neuron_idx] = resting_voltage

                spikes_per_neuron[neuron_idx] += 1
                current_bin = int(current_time // bin_rate)
                spikes_per_bin[current_bin] += 1

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
