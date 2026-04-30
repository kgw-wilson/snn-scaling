#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <signal.h>
#include <unistd.h>
#include <queue>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>

namespace py = pybind11;

volatile sig_atomic_t timed_out = 0;

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

class EventDrivenSimulation
{
public:
    torch::Tensor crow_indices;
    torch::Tensor col_indices;
    torch::Tensor weights;
    torch::Tensor delays;

    torch::Tensor membrane_voltages;
    torch::Tensor synaptic_currents;
    torch::Tensor last_update_times;
    torch::Tensor last_voltage_update_times;
    torch::Tensor last_spike_times;

    torch::Tensor spikes_per_neuron;
    torch::Tensor spikes_per_bin;

    EventQueue event_queue;

    int max_runtime;
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

    EventDrivenSimulation(
        torch::Tensor crow_indices,
        torch::Tensor col_indices,
        torch::Tensor weights,
        torch::Tensor delays,
        torch::Tensor membrane_voltages,
        torch::Tensor synaptic_currents,
        torch::Tensor last_update_times,
        torch::Tensor last_voltage_update_times,
        torch::Tensor last_spike_times,
        torch::Tensor spikes_per_neuron,
        torch::Tensor spikes_per_bin,
        int max_runtime,
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
        unsigned int seed)
        : crow_indices(crow_indices),
          col_indices(col_indices),
          weights(weights),
          delays(delays),
          membrane_voltages(membrane_voltages),
          synaptic_currents(synaptic_currents),
          last_update_times(last_update_times),
          last_voltage_update_times(last_voltage_update_times),
          last_spike_times(last_spike_times),
          spikes_per_neuron(spikes_per_neuron),
          spikes_per_bin(spikes_per_bin),
          max_runtime(max_runtime),
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
            event_queue.push({exp_dist(rng), i, poisson_weight, true});
    }

    py::dict run()
    {

        auto crow = crow_indices.accessor<int, 1>();
        auto col = col_indices.accessor<int, 1>();
        auto w = weights.accessor<float, 1>();
        auto d = delays.accessor<float, 1>();
        auto mv = membrane_voltages.accessor<float, 1>();
        auto sc = synaptic_currents.accessor<float, 1>();
        auto lut = last_update_times.accessor<float, 1>();
        auto lvut = last_voltage_update_times.accessor<float, 1>();
        auto lst = last_spike_times.accessor<float, 1>();
        auto spn = spikes_per_neuron.accessor<int, 1>();
        auto spb = spikes_per_bin.accessor<int, 1>();

        timed_out = 0;
        signal(SIGALRM, [](int)
               { timed_out = 1; });
        alarm(max_runtime);

        while (!event_queue.empty() && !timed_out)
        {
            Event e = event_queue.top();
            event_queue.pop();
            int i = e.neuron_idx;
            float current_time = e.time;

            if (e.time >= simulation_time)
                break;

            if (e.is_poisson)
                event_queue.push({current_time + exp_dist(rng), i, poisson_weight, true});

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
                    int start = crow[i];
                    int end = crow[i + 1];
                    for (int j = start; j < end; j++)
                        event_queue.push({current_time + d[j], col[j], w[j], false});

                    mv[i] = resting_voltage;
                    lst[i] = current_time;
                    lvut[i] = current_time + refractory_period;
                    spn[i]++;
                    spb[int(current_time / bin_rate)]++;
                }
            }
        }

        alarm(0);

        return py::dict(
            py::arg("spikes_per_neuron") = spikes_per_neuron,
            py::arg("spikes_per_bin") = spikes_per_bin,
            py::arg("timed_out") = (bool)timed_out);
    }
};

PYBIND11_MODULE(backend, m)
{
    py::class_<EventDrivenSimulation>(m, "EventDrivenSimulation")
        .def(py::init<
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 torch::Tensor,
                 int,
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
             py::arg("crow_indices"),
             py::arg("col_indices"),
             py::arg("weights"),
             py::arg("delays"),
             py::arg("membrane_voltages"),
             py::arg("synaptic_currents"),
             py::arg("last_update_times"),
             py::arg("last_voltage_update_times"),
             py::arg("last_spike_times"),
             py::arg("spikes_per_neuron"),
             py::arg("spikes_per_bin"),
             py::arg("max_runtime"),
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
        .def("run", &EventDrivenSimulation::run);
}