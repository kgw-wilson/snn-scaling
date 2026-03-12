import numpy as np
import torch

from shared.graph_creation import create_er_sparse_cpu
from shared.event_driven import EventQueue
from shared.monitoring import MonitoringWindow
from shared.simulation_config import ERGraphConfig, SNNConfig


def run_simulation_event_driven(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run an event-driven LIF spiking neural network simulation"""

    dtype = torch.float32

    # Unpack config objects to simplify simulation code and avoid attribute lookups
    num_neurons = graph_config.num_neurons
    (
        num_timesteps,
        timestep,
        membrane_time_constant,
        synaptic_time_constant,
        resting_voltage,
        threshold_voltage,
        poisson_prob,
    ) = (
        snn_config.num_timesteps,
        snn_config.timestep,
        snn_config.membrane_time_constant,
        snn_config.synaptic_time_constant,
        snn_config.resting_voltage,
        snn_config.threshold_voltage,
        snn_config.poisson_prob,
    )

    weights = create_er_sparse_cpu(config=graph_config, dtype=np.float32)

    event_queue = EventQueue()
    poisson_spikes_bool = torch.rand((num_timesteps, num_neurons)) < poisson_prob
    spike_indices = torch.nonzero(poisson_spikes_bool)
    events = [(t * timestep, n) for t, n in spike_indices.tolist()]
    event_queue.initialize_heap(events=events)

    membrane_voltages = torch.full((num_neurons,), resting_voltage, dtype=dtype)
    synaptic_currents = torch.zeros(num_neurons, dtype=dtype)
    last_update_times = torch.zeros(num_neurons, dtype=dtype)
    next_spike_times = torch.full((num_neurons,), float("inf"))

    with MonitoringWindow("simulation main loop"):

        while len(event_queue) > 0:

            current_time, neuron_idx = event_queue.pop()

            membrane_voltages[neuron_idx] = resting_voltage
            last_update_times[neuron_idx] = current_time

            connections_start = weights.indptr[neuron_idx]
            connections_end = weights.indptr[neuron_idx + 1]
            post_synaptic_indices = weights.indices[connections_start:connections_end]
            post_synaptic_weights = weights.data[connections_start:connections_end]
            post_synaptic_update_times = last_update_times[post_synaptic_indices]

            # Gather local copies only once to avoid slow advanced indexing
            post_i = synaptic_currents[post_synaptic_indices]
            post_v = membrane_voltages[post_synaptic_indices]

            # Unlike clock-driven simulations where voltages and currents update
            # every timestep, per-neuron update times are variable here
            time_deltas = current_time - post_synaptic_update_times
            synaptic_decays = torch.exp(-time_deltas / synaptic_time_constant)
            post_i = post_i * synaptic_decays + post_synaptic_weights
            membrane_decays = torch.exp(-time_deltas / membrane_time_constant)
            post_v = post_v * membrane_decays + post_i * (1 - membrane_decays)

            new_spikes_bool = post_v >= threshold_voltage
            new_spike_indices = post_synaptic_indices[new_spikes_bool]

            synaptic_currents[post_synaptic_indices] = post_i
            membrane_voltages[post_synaptic_indices] = post_v
            last_update_times[post_synaptic_indices] = current_time

            for new_spike_idx in new_spike_indices:

                new_spike_idx = int(new_spike_idx)
                new_spike_time = current_time + timestep

                # Check times of next scheduled spike for each neuron so queue does not
                # fill up with duplicate spikes (from the same neuron at the same time)
                if next_spike_times[new_spike_idx] > new_spike_time:
                    next_spike_times[new_spike_idx] = new_spike_time
                    event_queue.push((new_spike_time, new_spike_idx))
