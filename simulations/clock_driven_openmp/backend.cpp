#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <omp.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

volatile sig_atomic_t timed_out = 0;

class ClockDrivenOpenmpSimulation
{
public:
    int pt_num_threads;

    torch::Tensor crow_indices;
    torch::Tensor col_indices;
    torch::Tensor weights;
    torch::Tensor timestep_delays;
    torch::Tensor random_noise;
    torch::Tensor membrane_voltages;
    torch::Tensor synaptic_currents;
    torch::Tensor last_spike_times;
    torch::Tensor ring_buffer;
    torch::Tensor spikes_per_neuron;
    torch::Tensor spikes_per_bin;

    int max_runtime;
    int num_neurons;
    int num_timesteps;
    int num_bins;
    int buffer_size;
    int timesteps_per_bin;
    float timestep;
    float resistance;
    float poisson_weight;
    float poisson_prob;
    float refractory_period;
    float membrane_decay;
    float one_minus_decay;
    float synaptic_decay;
    float resting_voltage;
    float threshold_voltage;

    ClockDrivenOpenmpSimulation(
        torch::Tensor crow_indices,
        torch::Tensor col_indices,
        torch::Tensor weights,
        torch::Tensor timestep_delays,
        torch::Tensor random_noise,
        torch::Tensor membrane_voltages,
        torch::Tensor synaptic_currents,
        torch::Tensor last_spike_times,
        torch::Tensor ring_buffer,
        torch::Tensor spikes_per_neuron,
        torch::Tensor spikes_per_bin,
        int max_runtime,
        int num_neurons,
        int num_timesteps,
        int num_bins,
        int buffer_size,
        int timesteps_per_bin,
        float timestep,
        float resistance,
        float poisson_weight,
        float poisson_prob,
        float refractory_period,
        float membrane_decay,
        float synaptic_decay,
        float resting_voltage,
        float threshold_voltage)
        : crow_indices(crow_indices),
          col_indices(col_indices),
          weights(weights),
          timestep_delays(timestep_delays),
          random_noise(random_noise),
          membrane_voltages(membrane_voltages),
          synaptic_currents(synaptic_currents),
          last_spike_times(last_spike_times),
          ring_buffer(ring_buffer),
          spikes_per_neuron(spikes_per_neuron),
          spikes_per_bin(spikes_per_bin),
          max_runtime(max_runtime),
          num_neurons(num_neurons),
          num_timesteps(num_timesteps),
          num_bins(num_bins),
          buffer_size(buffer_size),
          timesteps_per_bin(timesteps_per_bin),
          timestep(timestep),
          resistance(resistance),
          poisson_weight(poisson_weight),
          poisson_prob(poisson_prob),
          refractory_period(refractory_period),
          membrane_decay(membrane_decay),
          one_minus_decay(1.0f - membrane_decay),
          synaptic_decay(synaptic_decay),
          resting_voltage(resting_voltage),
          threshold_voltage(threshold_voltage)
    {
        pt_num_threads = torch::get_num_threads();
    }

    py::dict run()
    {
        timed_out = 0;
        signal(SIGALRM, [](int)
               { timed_out = 1; });
        alarm(max_runtime);

        auto crow = crow_indices.accessor<int, 1>();
        auto col = col_indices.accessor<int, 1>();
        auto w = weights.accessor<float, 1>();
        auto d = timestep_delays.accessor<int, 1>();
        // auto mv = membrane_voltages.accessor<float, 1>();
        // auto sc = synaptic_currents.accessor<float, 1>();
        // auto lut = last_update_times.accessor<float, 1>();
        // auto lvut = last_voltage_update_times.accessor<float, 1>();
        // auto lst = last_spike_times.accessor<float, 1>();
        // auto spn = spikes_per_neuron.accessor<int, 1>();
        // auto spb = spikes_per_bin.accessor<int, 1>();

        int buffer_index = 0;

        for (int t = 0; t < num_timesteps && !timed_out; t++)
        {
            float current_time = t * timestep;
            buffer_index = (buffer_index + 1) % buffer_size;
            int bin_idx = t / timesteps_per_bin;

            random_noise.uniform_();

            synaptic_currents.mul_(synaptic_decay);
            synaptic_currents.add_(ring_buffer[buffer_index]);
            synaptic_currents.add_(poisson_weight * (random_noise < poisson_prob));

            torch::Tensor outside_refractory = (current_time - last_spike_times) >= refractory_period;

            torch::Tensor alpha = synaptic_currents * resistance + resting_voltage;
            torch::Tensor new_voltages = alpha * one_minus_decay + membrane_voltages * membrane_decay;
            membrane_voltages = torch::where(outside_refractory, new_voltages, membrane_voltages);

            torch::Tensor spikes_bool = membrane_voltages >= threshold_voltage;
            // Added
            torch::Tensor spiking_indices = spikes_bool.nonzero().squeeze(1).to(torch::kInt32);
            int n_spikes = spiking_indices.size(0);
            int *spiking_indices_ptr = spiking_indices.data_ptr<int32_t>();

            // for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++)
            // {
            //     ring_buffer[bucket_indices_in_buffer[t][bucket_idx]].add_(
            //         torch::mv(bucketized_weights[bucket_idx], spikes_bool.to(torch::kFloat32)));
            // }
            torch::set_num_threads(1);

#pragma omp parallel for schedule(static) num_threads(pt_num_threads)
            for (int src = 0; src < num_neurons; src++)
            {
                if (!spikes_bool[src].item<bool>())
                {
                    continue;
                }

                int start = crow[src];
                int end = crow[src + 1];

                for (int j = start; j < end; j++)
                {
                    int dst_idx = col[j];
                    int cur_buffer_idx = (buffer_index + d[j]) % buffer_size;
#pragma omp atomic
                    ring_buffer[cur_buffer_idx][dst_idx] += w[j];
                }
            }
            // #pragma omp parallel for schedule(static) num_threads(pt_num_threads)
            //             for (int i = 0; i < n_spikes; i++)
            //             {
            //                 int spiking_index = spiking_indices_ptr[i];
            //                 int start = crow[spiking_index];
            //                 int end = crow[spiking_index + 1];
            //                 for (int j = start; j < end; j++)
            //                 {
            //                     int target_idx = col[j];
            //                     int cur_buffer_idx = (buffer_index + d[j]) % buffer_size; // TODO: target_idx instead of j here?
            //                     ring_buffer[cur_buffer_idx][target_idx] += w[j];
            //                 }
            //             }

            torch::set_num_threads(pt_num_threads);

            ring_buffer[buffer_index].zero_();
            membrane_voltages = torch::where(spikes_bool, resting_voltage, membrane_voltages);
            last_spike_times = torch::where(spikes_bool, current_time, last_spike_times);

            spikes_per_neuron.add_(spikes_bool);
            spikes_per_bin[bin_idx].add_(spikes_bool.sum());
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
    py::class_<ClockDrivenOpenmpSimulation>(m, "ClockDrivenOpenmpSimulation")
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
                 float>(),
             py::arg("crow_indices"),
             py::arg("col_indices"),
             py::arg("weights"),
             py::arg("timestep_delays"),
             py::arg("random_noise"),
             py::arg("membrane_voltages"),
             py::arg("synaptic_currents"),
             py::arg("last_spike_times"),
             py::arg("ring_buffer"),
             py::arg("spikes_per_neuron"),
             py::arg("spikes_per_bin"),
             py::arg("max_runtime"),
             py::arg("num_neurons"),
             py::arg("num_timesteps"),
             py::arg("num_bins"),
             py::arg("buffer_size"),
             py::arg("timesteps_per_bin"),
             py::arg("timestep"),
             py::arg("resistance"),
             py::arg("poisson_weight"),
             py::arg("poisson_prob"),
             py::arg("refractory_period"),
             py::arg("membrane_decay"),
             py::arg("synaptic_decay"),
             py::arg("resting_voltage"),
             py::arg("threshold_voltage"))
        .def("run", &ClockDrivenOpenmpSimulation::run);
}
