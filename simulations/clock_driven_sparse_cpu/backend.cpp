#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_vsl.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

using CsrMatrix = std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<int>>;
using CsrMatrixList = std::vector<CsrMatrix>;

volatile sig_atomic_t timed_out = 0;

class ClockDrivenSparseCpuSimulation
{
public:
    std::vector<sparse_matrix_t> mkl_matrices;
    matrix_descr descr;

    torch::Tensor bucket_indices_in_buffer;

    torch::Tensor membrane_voltages;
    torch::Tensor synaptic_currents;
    torch::Tensor last_spike_times;
    torch::Tensor ring_buffer;
    torch::Tensor spikes_per_neuron;
    torch::Tensor spikes_per_bin;

    torch::Tensor spikes_float;
    torch::Tensor matmul_result;
    torch::Tensor random_noise;

    int max_runtime;
    int num_neurons;
    int num_timesteps;
    int num_buckets;
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

    ClockDrivenSparseCpuSimulation(
        CsrMatrixList bucketized_weights_csr,
        torch::Tensor bucket_indices_in_buffer,
        torch::Tensor membrane_voltages,
        torch::Tensor synaptic_currents,
        torch::Tensor last_spike_times,
        torch::Tensor ring_buffer,
        torch::Tensor spikes_per_neuron,
        torch::Tensor spikes_per_bin,
        int max_runtime,
        int num_neurons,
        int num_timesteps,
        int num_buckets,
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
        : bucket_indices_in_buffer(bucket_indices_in_buffer),
          membrane_voltages(membrane_voltages),
          synaptic_currents(synaptic_currents),
          last_spike_times(last_spike_times),
          ring_buffer(ring_buffer),
          spikes_per_neuron(spikes_per_neuron),
          spikes_per_bin(spikes_per_bin),
          max_runtime(max_runtime),
          num_neurons(num_neurons),
          num_timesteps(num_timesteps),
          num_buckets(num_buckets),
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
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        for (auto &[data, indices, indptr] : bucketized_weights_csr)
        {
            auto d = data.unchecked<1>();
            auto idx = indices.unchecked<1>();
            auto ptr = indptr.unchecked<1>();

            sparse_matrix_t mat;
            mkl_sparse_s_create_csr(
                &mat,
                SPARSE_INDEX_BASE_ZERO,
                num_neurons,
                num_neurons,
                const_cast<int *>(ptr.data(0)),
                const_cast<int *>(ptr.data(0)) + 1,
                const_cast<int *>(idx.data(0)),
                const_cast<float *>(d.data(0)));

            mkl_sparse_optimize(mat);
            mkl_matrices.push_back(mat);
        }

        spikes_float = torch::zeros({num_neurons});
        matmul_result = torch::zeros({num_neurons});
        random_noise = torch::empty({num_neurons});
        torch::Tensor all_results = torch::zeros({num_buckets, num_neurons});
    }

    ~ClockDrivenSparseCpuSimulation()
    {
        for (auto &mat : mkl_matrices)
            mkl_sparse_destroy(mat);
    }

    py::dict run()
    {
        timed_out = 0;
        signal(SIGALRM, [](int)
               { timed_out = 1; });
        alarm(max_runtime);

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
            spikes_float.copy_(spikes_bool.to(torch::kFloat32));

            // Sparse matmul per bucket into ring buffer
            auto bucket_indices_t = bucket_indices_in_buffer[t];
            for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++)
            {
                int target_idx = bucket_indices_t[bucket_idx].item<int>();

                mkl_sparse_s_mv(
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0f,
                    mkl_matrices[bucket_idx],
                    descr,
                    spikes_float.data_ptr<float>(),
                    0.0f,
                    matmul_result.data_ptr<float>());

                ring_buffer[target_idx].add_(matmul_result);
            }

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
    py::class_<ClockDrivenSparseCpuSimulation>(m, "ClockDrivenSparseCpuSimulation")
        .def(py::init<
                 CsrMatrixList,
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
             py::arg("bucketized_weights_csr"),
             py::arg("bucket_indices_in_buffer"),
             py::arg("membrane_voltages"),
             py::arg("synaptic_currents"),
             py::arg("last_spike_times"),
             py::arg("ring_buffer"),
             py::arg("spikes_per_neuron"),
             py::arg("spikes_per_bin"),
             py::arg("max_runtime"),
             py::arg("num_neurons"),
             py::arg("num_timesteps"),
             py::arg("num_buckets"),
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
        .def("run", &ClockDrivenSparseCpuSimulation::run);
}