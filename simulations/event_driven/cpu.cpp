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
    double time;
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
    py::array_t<double> delays;

    py::array_t<float> membrane_voltages;
    py::array_t<float> synaptic_currents;
    py::array_t<double> last_update_times;
    py::array_t<double> last_spike_times;

    py::array_t<int> spikes_per_neuron;
    py::array_t<int> spikes_per_bin;

    EventQueue event_queue;

    int num_neurons;
    int num_bins;
    double simulation_time;
    double poisson_rate;
    double refractory_period;
    double membrane_time_constant;
    double synaptic_time_constant;
    float resting_voltage;
    float threshold_voltage;
    double bin_rate;

    std::mt19937_64 rng;
    std::exponential_distribution<double> exp_dist;

    Simulation(
        py::array_t<int> indptr_arr,
        py::array_t<int> indices_arr,
        py::array_t<float> weights_arr,
        py::array_t<double> delays_arr,
        py::array_t<float> membrane_voltages_arr,
        py::array_t<float> synaptic_currents_arr,
        py::array_t<double> last_update_times_arr,
        py::array_t<double> last_spike_times_arr,
        py::array_t<int> spikes_per_neuron_arr,
        py::array_t<int> spikes_per_bin_arr,
        int num_neurons,
        int num_bins,
        double simulation_time,
        double poisson_rate,
        double refractory_period,
        double membrane_time_constant,
        double synaptic_time_constant,
        float resting_voltage,
        float threshold_voltage,
        double bin_rate,
        unsigned int seed) : indptr(indptr_arr),
                             indices(indices_arr),
                             weights(weights_arr),
                             delays(delays_arr),
                             membrane_voltages(membrane_voltages_arr),
                             synaptic_currents(synaptic_currents_arr),
                             last_update_times(last_update_times_arr),
                             last_spike_times(last_spike_times_arr),
                             spikes_per_neuron(spikes_per_neuron_arr),
                             spikes_per_bin(spikes_per_bin_arr),
                             num_neurons(num_neurons),
                             num_bins(num_bins),
                             simulation_time(simulation_time),
                             poisson_rate(poisson_rate),
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
            event_queue.push({exp_dist(rng), i, 0.0f, true});
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
        auto lst = last_spike_times.mutable_unchecked<1>();
        auto spn = spikes_per_neuron.mutable_unchecked<1>();
        auto spb = spikes_per_bin.mutable_unchecked<1>();

        while (!event_queue.empty())
        {
            Event e = event_queue.top();
            event_queue.pop();

            if (e.time >= simulation_time)
                break;

            bool can_spike = (lst[e.neuron_idx] + refractory_period <= e.time);

            if (e.is_poisson)
            {
                event_queue.push({e.time + exp_dist(rng), e.neuron_idx, 0.0f, true});

                if (can_spike)
                {
                    int start = ind[e.neuron_idx];
                    int end = ind[e.neuron_idx + 1];
                    for (int j = start; j < end; j++)
                    {
                        event_queue.push({e.time + d[j], idx[j], w[j], false});
                    }
                    mv[e.neuron_idx] = resting_voltage;
                    lst[e.neuron_idx] = e.time;
                    spn[e.neuron_idx]++;
                    spb[int(e.time / bin_rate)]++;
                }
            }
            else
            {
                double dt = e.time - lut[e.neuron_idx];
                float syn_decay = std::exp(-dt / synaptic_time_constant);
                float mem_decay = std::exp(-dt / membrane_time_constant);

                float new_current = sc[e.neuron_idx] * syn_decay + e.weight;
                float new_voltage = (mv[e.neuron_idx] - resting_voltage) * mem_decay + resting_voltage + new_current;

                sc[e.neuron_idx] = new_current;
                mv[e.neuron_idx] = new_voltage;
                lut[e.neuron_idx] = e.time;

                if (new_voltage >= threshold_voltage && can_spike)
                {
                    int start = ind[e.neuron_idx];
                    int end = ind[e.neuron_idx + 1];
                    for (int j = start; j < end; j++)
                    {
                        event_queue.push({e.time + d[j], idx[j], w[j], false});
                    }
                    mv[e.neuron_idx] = resting_voltage;
                    lst[e.neuron_idx] = e.time;
                    spn[e.neuron_idx]++;
                    spb[int(e.time / bin_rate)]++;
                }
            }
        }

        py::array_t<int> spn_out(num_neurons);
        py::array_t<int> spb_out(num_bins);
        std::copy(spn.data(0), spn.data(0) + num_neurons, spn_out.mutable_data());
        std::copy(spb.data(0), spb.data(0) + num_bins, spb_out.mutable_data());

        return py::dict(
            py::arg("spikes_per_neuron") = spn_out,
            py::arg("spikes_per_bin") = spb_out);
    }
};

PYBIND11_MODULE(cpu_cpp, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<
                 py::array_t<int>,
                 py::array_t<int>,
                 py::array_t<float>,
                 py::array_t<double>,
                 py::array_t<float>,
                 py::array_t<float>,
                 py::array_t<double>,
                 py::array_t<double>,
                 py::array_t<int>,
                 py::array_t<int>,
                 int,
                 int,
                 double,
                 double,
                 double,
                 double,
                 double,
                 float,
                 float,
                 double,
                 unsigned int>(),
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("weights"),
             py::arg("delays"),
             py::arg("membrane_voltages"),
             py::arg("synaptic_currents"),
             py::arg("last_update_times"),
             py::arg("last_spike_times"),
             py::arg("spikes_per_neuron"),
             py::arg("spikes_per_bin"),
             py::arg("num_neurons"),
             py::arg("num_bins"),
             py::arg("simulation_time"),
             py::arg("poisson_rate"),
             py::arg("refractory_period"),
             py::arg("membrane_time_constant"),
             py::arg("synaptic_time_constant"),
             py::arg("resting_voltage"),
             py::arg("threshold_voltage"),
             py::arg("bin_rate"),
             py::arg("seed"))
        .def("run", &Simulation::run);
}
