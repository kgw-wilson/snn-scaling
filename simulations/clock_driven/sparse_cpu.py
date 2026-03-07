import numpy as np

from shared.simulation_config import ERGraphConfig, SNNConfig
from shared.graph_creation import create_er_sparse_cpu
from shared.monitoring import MonitoringWindow


def run_simulation_sparse_cpu(graph_config: ERGraphConfig, snn_config: SNNConfig):
    """Run SNN simulation using sparse csr matrix for synaptic weights on CPU"""

    dtype = np.float32

    weights = create_er_sparse_cpu(config=graph_config, dtype=dtype)

    # Unpack config objects to simplify simulation code and avoid attribute lookups
    num_neurons = graph_config.num_neurons
    (
        num_timesteps,
        membrane_decay,
        synaptic_decay,
        resting_voltage,
        threshold_voltage,
        poisson_prob,
        poisson_weight,
    ) = (
        snn_config.num_timesteps,
        snn_config.membrane_decay,
        snn_config.synaptic_decay,
        snn_config.resting_voltage,
        snn_config.threshold_voltage,
        snn_config.poisson_prob,
        snn_config.poisson_weight,
    )

    membrane_voltages = np.full(num_neurons, resting_voltage, dtype=dtype)
    synaptic_currents = np.zeros(num_neurons, dtype=dtype)
    recurrent_spikes_bool = np.zeros(num_neurons, dtype=bool)
    recurrent_spikes_log = np.zeros(num_timesteps, dtype=dtype)
    poisson_spikes_log = np.zeros(num_timesteps, dtype=dtype)

    with MonitoringWindow("simulation main loop"):

        for t in range(num_timesteps):

            membrane_voltages = (
                membrane_voltages * membrane_decay
                + synaptic_currents * (1 - membrane_decay)
            )
            recurrent_spikes_bool = membrane_voltages >= threshold_voltage
            membrane_voltages[recurrent_spikes_bool] = resting_voltage

            recurrent_spikes_float = recurrent_spikes_bool.astype(dtype)
            poisson_spikes_bool = np.random.rand(num_neurons) < poisson_prob
            poisson_spikes_float = poisson_spikes_bool.astype(dtype)

            recurrent_spikes_log[t] = recurrent_spikes_bool.sum()
            poisson_spikes_log[t] = poisson_spikes_bool.sum()

            synaptic_currents = (
                synaptic_currents * synaptic_decay
                + weights.dot(recurrent_spikes_float)
                + poisson_weight * poisson_spikes_float
            )

    print("Mean recurrent spikes per timestep: ", recurrent_spikes_log.mean())
    print("Mean poisson spikes per timestep: ", poisson_spikes_log.mean())
