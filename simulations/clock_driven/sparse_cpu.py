import numpy as np

from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.graph_creation import create_er_sparse
from shared.monitoring import MonitoringWindow
from shared.reporting import report_spike_statistics


def run_simulation_sparse_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run SNN simulation using sparse csr matrix for synaptic weights on CPU"""

    np_dtype = np.float32

    weights = create_er_sparse(config=graph_config, use_numpy=True)

    # Unpack config objects to avoid attribute lookups
    num_neurons = graph_config.num_neurons
    timestep = snn_config.timestep
    resting_voltage = snn_config.resting_voltage
    threshold_voltage = snn_config.threshold_voltage
    num_timesteps = snn_config.num_timesteps
    membrane_decay = snn_config.membrane_decay
    synaptic_decay = snn_config.synaptic_decay
    poisson_prob = snn_config.poisson_prob
    bin_rate = snn_config.bin_rate
    num_bins = snn_config.num_bins

    membrane_voltages = np.full(num_neurons, resting_voltage, dtype=np_dtype)
    synaptic_currents = np.zeros(num_neurons, dtype=np_dtype)
    recurrent_spikes_bool = np.zeros(num_neurons, dtype=np.bool)
    poisson_spikes_bool = np.random.rand(num_timesteps, num_neurons) < poisson_prob

    spikes_per_neuron = np.zeros(num_neurons, dtype=np.int32)
    spikes_per_bin = np.zeros(num_bins, dtype=np.int32)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            membrane_voltages = (
                membrane_voltages * membrane_decay
                + synaptic_currents * (1 - membrane_decay)
            )

            combined_spikes_bool = np.logical_or(
                poisson_spikes_bool[t], recurrent_spikes_bool
            )
            combined_spikes_float = combined_spikes_bool.astype(np_dtype)

            synaptic_currents = (
                synaptic_currents * synaptic_decay
                + weights.dot(combined_spikes_float)
            )

            recurrent_spikes_bool = membrane_voltages >= threshold_voltage
            membrane_voltages[recurrent_spikes_bool] = resting_voltage

            spikes_per_neuron += combined_spikes_bool
            current_time = t * timestep
            current_bin = int(current_time // bin_rate)
            spikes_per_bin[current_bin] += combined_spikes_bool.sum()

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)
