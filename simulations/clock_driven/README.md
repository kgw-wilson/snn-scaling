# Clock-driven Simiulations

The simulations in this folder are all clock-driven (i.e. neuron events happen according to a fixed timestep).

- `dense.py` is a simulation using PyTorch that can run on both CPU and GPU.

- `sparse_cpu.py` is a simulation using numpy and scipy that can run on CPU only. numpy and scipy are used because PyTorch operations for sparse operations on the CPU is very slow as of 03/2026.

- `sparse_gpu.py` is a simulation using PyTorch that can run with CUDA only. PyTorch does not support sparse matrix operations on MPS (Apple Silicon) as of 03/2026.

Main simulation loops in this folder follow this flow: update currents, update voltages, find spiking neurons, schdule current in the future, perform necessary resets, and then do reporting. This order is used for correctness and to maintain biological plausibility.