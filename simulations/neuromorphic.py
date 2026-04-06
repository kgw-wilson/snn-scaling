from pyNN.random import RandomDistribution
import pyNN.spiNNaker as p
from spynnaker.pyNN.models.neuron.builds import IFCurrExpBase
from shared.monitoring import MonitoringWindow
from shared.simulation_config import SimulationConfig
from shared.reporting import create_spike_reporting_tensors, report_spike_statistics


def neuromorphic(sim_config: SimulationConfig, seed: int):
    """Run SNN using SpyNNaker to access neuromorphic hardware"""

    p.setup(timestep=sim_config.timestep * 1000, time_scale_factor=1000)
    p.set_number_of_neurons_per_core(p.IF_curr_exp, 64)
    p.set_number_of_neurons_per_core(p.SpikeSourcePoisson, 64)

    cellclass_instance = IFCurrExpBase(
        tau_m=sim_config.membrane_time_constant * 1000,
        cm=sim_config.capacitance * 1000,
        v_rest=sim_config.resting_voltage * 1000,
        v_reset=sim_config.resting_voltage * 1000,
        v_thresh=sim_config.threshold_voltage * 1000,
        tau_syn_E=sim_config.synaptic_time_constant * 1000,
        tau_syn_I=0.0,
        tau_refrac=sim_config.refractory_period * 1000,
        i_offset=0.0,
    )

    pop_exc = p.Population(
        size=sim_config.num_neurons,
        cellclass=cellclass_instance,
        initial_values={"v": sim_config.resting_voltage * 1000, "isyn_exc": 0.0},
        label="neurons",
        seed=seed,
    )

    stim_exc = p.Population(
        size=sim_config.num_neurons,
        cellclass=p.SpikeSourcePoisson(rate=sim_config.poisson_rate),
        seed=seed,
    )

    p.Projection(
        pop_exc,
        pop_exc,
        p.FixedProbabilityConnector(0.1),
        p.StaticSynapse(
            weight=sim_config.recurrent_weight,
            delay=RandomDistribution(
                "uniform",
                (sim_config.min_delay * 1000, sim_config.max_delay * 1000),
            ),
        ),
        receptor_type="excitatory",
    )

    p.Projection(
        stim_exc,
        pop_exc,
        p.OneToOneConnector(),
        p.StaticSynapse(weight=sim_config.poisson_weight, delay=0.0),
        receptor_type="excitatory",
    )

    pop_exc.initialize(v=sim_config.resting_voltage)

    pop_exc.record("spikes")

    with MonitoringWindow("simulation main"):
        p.run(sim_config.simulation_time * 1000)

    data = pop_exc.get_data("spikes")

    spikes_per_neuron, spikes_per_bin = create_spike_reporting_tensors(sim_config)

    for i, spiketrain in enumerate(data.segments[0].spiketrains):
        spikes_per_neuron[i] = spiketrain.sum().item()
        for spike_time in spiketrain:
            bin_idx = int(spike_time / sim_config.bin_rate)  # TODO: what time units?
            spikes_per_bin[bin_idx] += 1

    report_spike_statistics(spikes_per_neuron, spikes_per_bin)

    p.end()
