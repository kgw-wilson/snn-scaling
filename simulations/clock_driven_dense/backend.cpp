#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

volatile sig_atomic_t timed_out = 0;

class ClockDrivenDenseSimulation
{
public:
    torch::Tensor bucket_indices_in_buffer;
    torch::Tensor bucketized_weights;
    torch::Tensor random_noise;
    torch::Tensor membrane_voltages;
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
    float weight;
    float poisson_prob;
    float refractory_period;
    float membrane_decay;
    float resting_voltage;
    float threshold_voltage;

    ClockDrivenDenseSimulation(
        torch::Tensor bucket_indices_in_buffer,
        torch::Tensor bucketized_weights,
        torch::Tensor random_noise,
        torch::Tensor membrane_voltages,
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
        float weight,
        float poisson_prob,
        float refractory_period,
        float membrane_decay,
        float resting_voltage,
        float threshold_voltage)
        : bucket_indices_in_buffer(bucket_indices_in_buffer),
          bucketized_weights(bucketized_weights),
          random_noise(random_noise),
          membrane_voltages(membrane_voltages),
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
          weight(weight),
          poisson_prob(poisson_prob),
          refractory_period(refractory_period),
          membrane_decay(membrane_decay),
          resting_voltage(resting_voltage),
          threshold_voltage(threshold_voltage)
    {
    }

    py::dict run()
    {
        timed_out = 0;
        signal(SIGALRM, [](int)
               { timed_out = 1; });
        alarm(max_runtime);

        int buffer_index = 0;

        torch::Tensor new_voltages = torch::empty_like(membrane_voltages);

        for (int t = 0; t < num_timesteps && !timed_out; t++)
        {
            float current_time = t * timestep;
            buffer_index = (buffer_index + 1) % buffer_size;
            int bin_idx = t / timesteps_per_bin;

            // random_noise.uniform_();

            random_noise.fill_(poisson_prob);                            // added
            torch::Tensor poisson_counts = torch::poisson(random_noise); // added
            
            // std::cout << poisson_counts.sum().item<float>() << " spikes from Poisson input at time " << current_time << std::endl; // added

            torch::Tensor outside_refractory = (current_time - last_spike_times) >= refractory_period;

            new_voltages = resting_voltage + (membrane_voltages - resting_voltage) * membrane_decay;
            new_voltages.add_(ring_buffer[buffer_index]);
            // new_voltages.add_(weight * (random_noise < poisson_prob));
            new_voltages.add_(weight * poisson_counts); // added

            membrane_voltages = torch::where(outside_refractory, new_voltages, membrane_voltages);

            torch::Tensor spikes_bool = membrane_voltages >= threshold_voltage;

            ring_buffer.index_add_(
                0,
                bucket_indices_in_buffer[t],
                bucketized_weights.matmul(spikes_bool.to(torch::kFloat32)));

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
    py::class_<ClockDrivenDenseSimulation>(m, "ClockDrivenDenseSimulation")
        .def(py::init<
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
                 float>(),
             py::arg("bucket_indices_in_buffer"),
             py::arg("bucketized_weights"),
             py::arg("random_noise"),
             py::arg("membrane_voltages"),
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
             py::arg("weight"),
             py::arg("poisson_prob"),
             py::arg("refractory_period"),
             py::arg("membrane_decay"),
             py::arg("resting_voltage"),
             py::arg("threshold_voltage"))
        .def("run", &ClockDrivenDenseSimulation::run);
}
