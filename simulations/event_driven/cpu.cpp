#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

namespace py = pybind11;

struct Event
{
    float time;
    int neuron_idx;
    float weight;
    bool is_poisson;

    bool operator>(const Event &other) const
    {
        return time > other.time;
    }
};

using EventQueue = std::priority_queue<Event, std::vector<Event>, std::greater<Event>>;

class Simulation
{
public:
    py::array_t<int> indptr;
    py::array_t<int> indices;
    py::array_t<float> weights;
    py::array_t<float> delays;

    py::array_t<float> membrane_voltages;
    py::array_t<float> synaptic_currents;
    py::array_t<float> last_update_times;
    py::array_t<float> last_voltage_update_times;
    py::array_t<float> last_spike_times;

    py::array_t<int> spikes_per_neuron;
    py::array_t<int> spikes_per_bin;

    EventQueue event_queue;

    int num_neurons;
    int num_bins;
    float resistance;
    float simulation_time;
    float poisson_rate;
    float poisson_weight;
    float refractory_period;
    float membrane_time_constant;
    float synaptic_time_constant;
    float resting_voltage;
    float threshold_voltage;
    float bin_rate;

    std::mt19937 rng;
    std::exponential_distribution<float> exp_dist;

    Simulation(
        py::array_t<int> indptr_arr,
        py::array_t<int> indices_arr,
        py::array_t<float> weights_arr,
        py::array_t<float> delays_arr,
        py::array_t<float> membrane_voltages_arr,
        py::array_t<float> synaptic_currents_arr,
        py::array_t<float> last_update_times_arr,
        py::array_t<float> last_voltage_update_times_arr,
        py::array_t<float> last_spike_times_arr,
        py::array_t<int> spikes_per_neuron_arr,
        py::array_t<int> spikes_per_bin_arr,
        int num_neurons,
        int num_bins,
        float resistance,
        float simulation_time,
        float poisson_rate,
        float poisson_weight,
        float refractory_period,
        float membrane_time_constant,
        float synaptic_time_constant,
        float resting_voltage,
        float threshold_voltage,
        float bin_rate,
        unsigned int seed) : indptr(indptr_arr),
                             indices(indices_arr),
                             weights(weights_arr),
                             delays(delays_arr),
                             membrane_voltages(membrane_voltages_arr),
                             synaptic_currents(synaptic_currents_arr),
                             last_update_times(last_update_times_arr),
                             last_voltage_update_times(last_voltage_update_times_arr),
                             last_spike_times(last_spike_times_arr),
                             spikes_per_neuron(spikes_per_neuron_arr),
                             spikes_per_bin(spikes_per_bin_arr),
                             num_neurons(num_neurons),
                             num_bins(num_bins),
                             resistance(resistance),
                             simulation_time(simulation_time),
                             poisson_rate(poisson_rate),
                             poisson_weight(poisson_weight),
                             refractory_period(refractory_period),
                             membrane_time_constant(membrane_time_constant),
                             synaptic_time_constant(synaptic_time_constant),
                             resting_voltage(resting_voltage),
                             threshold_voltage(threshold_voltage),
                             bin_rate(bin_rate),
                             rng(seed),
                             exp_dist(poisson_rate)
    {
        for (int i = 0; i < num_neurons; i++)
        {
            event_queue.push({exp_dist(rng), i, poisson_weight, true});
        }
    }

    py::dict run()
    {
        auto ind = indptr.unchecked<1>();
        auto idx = indices.unchecked<1>();
        auto w = weights.unchecked<1>();
        auto d = delays.unchecked<1>();
        auto mv = membrane_voltages.mutable_unchecked<1>();
        auto sc = synaptic_currents.mutable_unchecked<1>();
        auto lut = last_update_times.mutable_unchecked<1>();
        auto lvut = last_voltage_update_times.mutable_unchecked<1>();
        auto lst = last_spike_times.mutable_unchecked<1>();
        auto spn = spikes_per_neuron.mutable_unchecked<1>();
        auto spb = spikes_per_bin.mutable_unchecked<1>();

        while (!event_queue.empty())
        {
            Event e = event_queue.top();
            event_queue.pop();
            int i = e.neuron_idx;
            float current_time = e.time;

            if (e.time >= simulation_time)
                break;

            if (e.is_poisson)
            {
                event_queue.push({current_time + exp_dist(rng), i, poisson_weight, true});
            }

            bool outside_refractory = (current_time - lst[i]) >= refractory_period;

            float dt = e.time - lut[i];
            float syn_decay = std::exp(-dt / synaptic_time_constant);
            float new_current = sc[i] * syn_decay + e.weight;

            sc[i] = new_current;
            lut[i] = current_time;

            if (outside_refractory)
            {
                float v_dt = current_time - lvut[i];
                float mem_decay = std::exp(-v_dt / membrane_time_constant);
                float alpha = new_current * resistance + resting_voltage;
                float new_voltage = alpha + (mv[i] - alpha) * mem_decay;

                mv[i] = new_voltage;
                lvut[i] = current_time;

                if (new_voltage >= threshold_voltage)
                {
                    int start = ind[i];
                    int end = ind[i + 1];
                    for (int j = start; j < end; j++)
                    {
                        event_queue.push({current_time + d[j], idx[j], w[j], false});
                    }
                    mv[i] = resting_voltage;
                    lst[i] = current_time;
                    lvut[i] = current_time + refractory_period - 0.1e-3; // May need to slightly adjust (with -timestep) to exactly match clock-driven
                    spn[i]++;
                    spb[int(current_time / bin_rate)]++;
                }
            }
        }

        return py::dict(
            py::arg("spikes_per_neuron") = spikes_per_neuron,
            py::arg("spikes_per_bin") = spikes_per_bin);
    }
};

PYBIND11_MODULE(cpu_cpp, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<
                 py::array_t<int>,
                 py::array_t<int>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<int>,
                 py::array_t<int>,
                 int,
                 int,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float,
                 float,
                 unsigned int>(),
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("weights"),
             py::arg("delays"),
             py::arg("membrane_voltages"),
             py::arg("synaptic_currents"),
             py::arg("last_update_times"),
             py::arg("last_voltage_update_times"),
             py::arg("last_spike_times"),
             py::arg("spikes_per_neuron"),
             py::arg("spikes_per_bin"),
             py::arg("num_neurons"),
             py::arg("num_bins"),
             py::arg("resistance"),
             py::arg("simulation_time"),
             py::arg("poisson_rate"),
             py::arg("poisson_weight"),
             py::arg("refractory_period"),
             py::arg("membrane_time_constant"),
             py::arg("synaptic_time_constant"),
             py::arg("resting_voltage"),
             py::arg("threshold_voltage"),
             py::arg("bin_rate"),
             py::arg("seed"))
        .def("run", &Simulation::run);
}
