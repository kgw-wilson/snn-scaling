"""
TODO: add in helper functions used in clock-driven and move them
to some shared place
TODO: look at runtime and python overhead and consider adding
numba
Your phase diagram becomes: for a given (N, p, firing rate) point, which architecture
wins on energy per SynOp? With topology as a qualitative modifier. That's a rich enough
parameter space to tell a compelling story without being intractable to run.
"""

import math
import random
import numpy as np
import torch

from shared.graph_creation import create_er_sparse
from shared.event_driven import EventQueue
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.utils import create_state_variables


def event_driven_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run an event-driven LIF spiking neural network simulation"""

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    simulation_time = snn_config.simulation_time
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay
    membrane_time_constant = snn_config.membrane_time_constant
    synaptic_time_constant = snn_config.synaptic_time_constant
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    poisson_rate = snn_config.poisson_rate
    refractory_period = snn_config.refractory_period
    bin_rate = snn_config.bin_rate

    weights = create_er_sparse(config=graph_config, use_numpy=True)
    indptr = weights.indptr.tolist()
    indices = weights.indices.tolist()
    data = weights.data.tolist()
    delays = [random.uniform(min_delay, max_delay) for _ in data]

    event_queue = EventQueue()
    events = []

    for neuron_idx in range(num_neurons):
        events.append((random.expovariate(lambd=poisson_rate), neuron_idx, 0.0, True))

    event_queue.initialize_heap(events=events)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config, snn_config
    )
    last_update_times = torch.full((num_neurons,), -torch.inf)

    membrane_voltages = membrane_voltages.tolist()
    synaptic_currents = synaptic_currents.tolist()
    last_spike_times = last_spike_times.tolist()
    last_update_times = last_update_times.tolist()

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config, snn_config
    )

    with MonitoringWindow("simulation main loop"):

        while len(event_queue) > 0:

            current_time, neuron_idx, weight, is_poisson = event_queue.pop()

            if current_time >= simulation_time:
                break

            can_spike = (
                last_update_times[neuron_idx] + refractory_period <= current_time
            )

            if is_poisson:

                time_to_next_spike = random.expovariate(lambd=poisson_rate)
                time_of_next_spike = current_time + time_to_next_spike  # + refractory?
                event_queue.push((time_of_next_spike, neuron_idx, 0.0, True))

                if can_spike:
                    connections_start = indptr[neuron_idx]
                    connections_end = indptr[neuron_idx + 1]
                    post_synaptic_indices = indices[connections_start:connections_end]
                    post_synaptic_weights = data[connections_start:connections_end]
                    post_synaptic_delays = delays[connections_start:connections_end]

                    for target_idx, weight, delay in zip(
                        post_synaptic_indices,
                        post_synaptic_weights,
                        post_synaptic_delays,
                    ):
                        event_queue.push(
                            (
                                current_time + delay,
                                target_idx,
                                weight,
                                False,
                            )
                        )

                    membrane_voltages[neuron_idx] = resting_voltage

                    spikes_per_neuron[neuron_idx] += 1
                    current_bin = int(current_time // bin_rate)
                    spikes_per_bin[current_bin] += 1

            else:

                time_since_last_update = current_time - last_update_times[neuron_idx]
                synaptic_decay = math.exp(
                    -time_since_last_update / synaptic_time_constant
                )
                membrane_decay = math.exp(
                    -time_since_last_update / membrane_time_constant
                )

                updated_current = (
                    synaptic_currents[neuron_idx] * synaptic_decay + weight
                )
                updated_voltage = (
                    (membrane_voltages[neuron_idx] - resting_voltage) * membrane_decay
                    + resting_voltage
                    + updated_current
                )

                synaptic_currents[neuron_idx] = updated_current
                membrane_voltages[neuron_idx] = updated_voltage
                last_update_times[neuron_idx] = current_time

                if updated_voltage >= threshold_voltage and can_spike:

                    connections_start = indptr[neuron_idx]
                    connections_end = indptr[neuron_idx + 1]
                    post_synaptic_indices = indices[connections_start:connections_end]
                    post_synaptic_weights = data[connections_start:connections_end]
                    post_synaptic_delays = delays[connections_start:connections_end]

                    for target_idx, weight, delay in zip(
                        post_synaptic_indices,
                        post_synaptic_weights,
                        post_synaptic_delays,
                    ):
                        event_queue.push(
                            (
                                current_time + delay,
                                target_idx,
                                weight,
                                False,
                            )
                        )

                    membrane_voltages[neuron_idx] = resting_voltage

                    spikes_per_neuron[neuron_idx] += 1
                    current_bin = int(current_time // bin_rate)
                    spikes_per_bin[current_bin] += 1

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
