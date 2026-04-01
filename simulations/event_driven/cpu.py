"""
TODO: add in helper functions used in clock-driven and move them
to some shared place
TODO: look at runtime and python overhead and consider adding
numba
Your phase diagram becomes: for a given (N, p, firing rate) point, which architecture
wins on energy per SynOp? With topology as a qualitative modifier. That's a rich enough
parameter space to tell a compelling story without being intractable to run.
"""

import numpy as np
import torch

from shared.graph_creation import create_er_sparse
from shared.event_driven import EventQueue
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics
from shared.simulation_config import ERGraphConfig, SNNConfig


def event_driven_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run an event-driven LIF spiking neural network simulation"""

    weights = create_er_sparse(config=graph_config, use_numpy=True)
    indptr = weights.indptr  # row pointer (same as crow)
    indices = weights.indices  # column indices (same as cols)
    data = weights.data

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    simulation_time = snn_config.simulation_time
    num_timesteps = snn_config.num_timesteps
    timestep = snn_config.timestep
    # delay = snn_config.delay
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay
    membrane_time_constant = snn_config.membrane_time_constant
    synaptic_time_constant = snn_config.synaptic_time_constant
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    poisson_rate = snn_config.poisson_rate
    poisson_prob = snn_config.poisson_prob
    refractory_period = snn_config.refractory_period
    bin_rate = snn_config.bin_rate
    num_bins = snn_config.num_bins

    # delays = torch.empty(num_neurons, num_neurons, device=device, dtype=dtype)
    # delays.uniform_(min_delay, max_delay)
    delays = np.random.uniform(
        low=min_delay, high=max_delay, size=(num_neurons, num_neurons)
    )

    event_queue = EventQueue()
    events = []
    poisson_spike_times = np.random.exponential(
        scale=1.0 / poisson_rate, size=num_neurons
    )

    for neuron_idx in range(num_neurons):

        # connections_start = indptr?[neuron_idx]  # .item()
        # connections_end = indptr[ne>uron_idx + 1]  # .item()
        # post_synaptic_indices = indices[connections_start:connections_end]
        # post_synaptic_weights = data[connections_start:connections_end]
        spike_time = float(poisson_spike_times[neuron_idx])
        weight = 0.0
        is_poisson = True

        events.append((spike_time, neuron_idx, weight, is_poisson))

    event_queue.initialize_heap(events=events)

    membrane_voltages = torch.full((num_neurons,), resting_voltage, dtype=dtype)
    synaptic_currents = torch.zeros(num_neurons, dtype=dtype)

    last_update_times = torch.full((num_neurons,), -torch.inf)

    spikes_per_neuron = torch.zeros(num_neurons)
    spikes_per_bin = torch.zeros(num_bins)

    with MonitoringWindow("simulation main loop"):

        while len(event_queue) > 0:

            current_time, neuron_idx, weight, is_poisson = event_queue.pop()

            # print(f"{neuron_idx=}")

            if current_time >= simulation_time:
                # raise RuntimeError(f"Hit end at {current_time}")
                break
            can_spike = (
                last_update_times[neuron_idx] + refractory_period <= current_time
            )

            if is_poisson:

                time_to_next_spike = np.random.exponential(
                    scale=1.0 / poisson_rate,
                )
                time_of_next_spike = current_time + time_to_next_spike # + refractory?
                event_queue.push(
                    (float(time_of_next_spike), int(neuron_idx), 0.0, is_poisson)
                )

                if can_spike:
                    connections_start = indptr[neuron_idx]
                    connections_end = indptr[neuron_idx + 1]
                    post_synaptic_indices = indices[connections_start:connections_end]
                    post_synaptic_weights = data[connections_start:connections_end]

                    for target_idx, connection_weight in zip(
                        post_synaptic_indices, post_synaptic_weights
                    ):
                        arrival_time = float(
                            current_time + delays[neuron_idx, target_idx] # + refractory?
                        )
                        event_queue.push(
                            (
                                arrival_time,
                                int(target_idx),
                                float(connection_weight),
                                False,
                            )
                        )

                    membrane_voltages[neuron_idx] = resting_voltage

                    spikes_per_neuron[neuron_idx] += 1
                    current_bin = int(current_time // bin_rate)
                    spikes_per_bin[current_bin] += 1

            else:

                time_since_last_update = current_time - last_update_times[neuron_idx]
                synaptic_decay = torch.exp(
                    -time_since_last_update / synaptic_time_constant
                )
                membrane_decay = torch.exp(
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

                    for target_idx, connection_weight in zip(
                        post_synaptic_indices, post_synaptic_weights
                    ):
                        arrival_time = float(
                            current_time + delays[neuron_idx, target_idx] # + refractory?
                        )
                        event_queue.push(
                            (
                                arrival_time,
                                int(target_idx),
                                float(connection_weight),
                                False,
                            )
                        )

                    membrane_voltages[neuron_idx] = resting_voltage

                    spikes_per_neuron[neuron_idx] += 1
                    current_bin = int(current_time // bin_rate)
                    spikes_per_bin[current_bin] += 1

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
