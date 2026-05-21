import torch
import brian2 as b2
from shared.monitoring import MonitoringWindow
from shared.reporting import report_statistics
from shared.simulation_config import SimulationConfig


def clock_driven_brian2(sim_config: SimulationConfig, seed: int) -> None:
    """Run clock-driven SNN simulation in Brian2 using voltage-only LIF updates."""

    b2.seed(seed)
    b2.start_scope()

    b2.defaultclock.dt = sim_config.timestep * b2.second

    tau_membrane = sim_config.membrane_time_constant * b2.second
    v_threshold = sim_config.threshold_voltage * b2.volt
    v_reset = sim_config.resting_voltage * b2.volt
    J = sim_config.weight * b2.volt
    J_inh = sim_config.inhibitory_weight_multiplier * J
    min_delay = sim_config.min_delay * b2.second
    max_delay = sim_config.max_delay * b2.second

    neurons = b2.NeuronGroup(
        N=sim_config.num_neurons,
        model="dv/dt = (v_reset - v) / tau_membrane : volt (unless refractory)",
        threshold="v >= v_threshold",
        reset="v = v_reset",
        refractory=sim_config.refractory_period * b2.second,
        method="exact",
    )
    neurons.v = v_reset

    excitatory_neurons = neurons[: sim_config.num_excitatory]
    inhibitory_neurons = neurons[sim_config.num_excitatory :]

    exc_synapses = b2.Synapses(excitatory_neurons, target=neurons, on_pre="v += J")
    exc_synapses.connect(p=sim_config.connection_prob)
    exc_synapses.delay = "rand() * (max_delay - min_delay) + min_delay"

    inh_synapses = b2.Synapses(inhibitory_neurons, target=neurons, on_pre="v += J_inh")
    inh_synapses.connect(p=sim_config.connection_prob)
    inh_synapses.delay = "rand() * (max_delay - min_delay) + min_delay"

    external_poisson_input = b2.PoissonInput(
        target=neurons,
        target_var="v",
        N=int(sim_config.num_excitatory * sim_config.connection_prob),
        rate=sim_config.poisson_rate * b2.Hz,
        weight=J,
    )

    spike_monitor = b2.SpikeMonitor(neurons, record=False)

    net = b2.Network(
        neurons, exc_synapses, inh_synapses, external_poisson_input, spike_monitor
    )

    with MonitoringWindow("Simulation main") as monitor:
        net.run(sim_config.simulation_time * b2.second)

    report_statistics(
        sim_config,
        "clock_driven_brian2",
        monitor.elapsed_time,
        torch.tensor(spike_monitor.count, dtype=torch.float32),
        torch.tensor([], dtype=torch.float32),
    )
