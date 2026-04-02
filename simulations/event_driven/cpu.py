import numpy as np
import torch

from shared.graph_creation import create_er_sparse
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics, create_spike_reporting_tensors
from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.utils import create_state_variables
import simulations.event_driven.cpu_cpp as cpu_cpp


# TODO: remove default seed
def event_driven_cpu(
    graph_config: ERGraphConfig, snn_config: SNNConfig, seed: int
) -> None:
    """Run an event-driven LIF spiking neural network simulation"""

    np.random.seed(seed)

    dtype = graph_config.dtype
    min_delay = snn_config.min_delay
    max_delay = snn_config.max_delay

    weights = create_er_sparse(config=graph_config, use_numpy=True)
    delays = np.random.uniform(
        low=min_delay, high=max_delay, size=len(weights.data)
    ).astype(np.float64)

    membrane_voltages, synaptic_currents, last_spike_times = create_state_variables(
        graph_config, snn_config
    )
    last_update_times = np.full(graph_config.num_neurons, -np.inf, dtype=np.float64)
    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(
        graph_config, snn_config
    )

    sim = cpu_cpp.Simulation(
        indptr=weights.indptr.astype(np.int32),
        indices=weights.indices.astype(np.int32),
        weights=weights.data.astype(np.float32),
        delays=delays,
        membrane_voltages=membrane_voltages.numpy().astype(np.float32),
        synaptic_currents=synaptic_currents.numpy().astype(np.float32),
        last_update_times=last_update_times,
        last_spike_times=last_spike_times.numpy().astype(np.float64),
        spikes_per_neuron=spikes_per_neuron.numpy().astype(np.int32),
        spikes_per_bin=spikes_per_bin.numpy().astype(np.int32),
        num_neurons=graph_config.num_neurons,
        num_bins=snn_config.num_bins,
        simulation_time=snn_config.simulation_time,
        poisson_rate=snn_config.poisson_rate,
        refractory_period=snn_config.refractory_period,
        membrane_time_constant=snn_config.membrane_time_constant,
        synaptic_time_constant=snn_config.synaptic_time_constant,
        resting_voltage=snn_config.resting_voltage,
        threshold_voltage=snn_config.threshold_voltage,
        bin_rate=snn_config.bin_rate,
        seed=seed,
    )

    with MonitoringWindow("simulation main loop"):

        result = sim.run()

    report_spike_statistics(
        torch.tensor(result["spikes_per_neuron"], dtype=dtype),
        torch.tensor(result["spikes_per_bin"], dtype=dtype),
    )
