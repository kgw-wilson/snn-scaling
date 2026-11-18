# Claims and sources for snn_scaling_talk.pptx

Every number in the deck, with the file or commit it comes from. "Derived" means simple arithmetic on logged values (done in `extract_data.py`, stored under `derived` in `deck_data.json`). "Analytic" means computed from the code, not measured. "External" means it is not in the repo.

Abbreviations: **04-30** = `logs/04-30/results.csv` (main sweep, commit 63ab99a code, Xeon E5-2696 v4 + RTX 4070 Ti Super per README, rented on Vast.ai per the author; the provider is not recorded in the repo). **HEAD csv** = `results.csv` at repo root (commit eb02eb9).

## Setup and model (slides 4–7, 10–14)

| Claim | Source |
|---|---|
| α = exp(−Δt/τ), exact update (not Euler) | `shared/simulation_config.py` `membrane_decay`; `simulations/README.md` ("analytical form instead of Euler") |
| HEAD dense: V ← V_rest + (V − V_rest)·α + ring + J·Poisson | `simulations/clock_driven_dense/backend.cpp` lines 106–118 |
| Cutoff ≈ 16 Hz for τ = 10 ms; α = 0.905 (1 ms), 0.990 (0.1 ms); β = 0.819 | Derived: 1/(2π·0.01), e^−0.1, e^−0.01, e^−0.2 |
| Apr 30 model: two-pole, τs = 5 ms, τm = RC = 10 Ω·1 mF = 10 ms | `git show 63ab99a:simulations/run_all.py` (resistance 10.0, capacitance 1e-3, synaptic_time_constant 5e-3); `63ab99a:shared/simulation_config.py` |
| Sweep: N ∈ {10,…,10⁴}, 9 values of p, Δt = 1 ms, T = 1 s, delays 2–4 ms, refractory 20 ms, Poisson 50 Hz, threshold 20 mV, 1 repeat, 300 s timeout | `git show 63ab99a:simulations/run_all.py` |
| Sweep is all-excitatory | `git show 63ab99a:shared/graph_creation.py` (no inhibitory scaling) |
| Exactly ⌊N²p⌋ edges | `shared/graph_creation.py` (`num_edges = int(N*N*p)`, randperm) |
| Brian2 setup: N = 1000, p ∈ {0.01,0.1,0.3}, Δt = 0.1 ms, 20% inhibitory, g = −5, J = 0.1 mV, θ = 10 mV, τ = 10 ms, refractory 2 ms, η = 2, 3 repeats | `simulations/run_all.py` (HEAD), `simulations/brian2/runner.py`, `shared/simulation_config.py` (`poisson_rate`, `poisson_prob`) |
| B = 3 buckets at Δt = 1 ms; 21 at 0.1 ms; ring buffer 5 / 41 slots | Analytic from `shared/clock_driven.py` (`_compute_delay_buckets`, `create_ring_buffer`) |
| Bucket 2 always empty for d ~ U(2,4) ms | Analytic: `torch.bucketize(..., right=True) - 1` with boundaries [2,3,4] ms |
| Read index advanced before read, so delivery was one step early (Apr 30 code and eb02eb9) | `git show eb02eb9:simulations/clock_driven_dense/backend.cpp` lines 94–127 (sparse: lines 108–136; same at 63ab99a) vs `create_lookup_tensors` offsets |
| Fixed: `buffer_index = t % buffer_size` in dense and sparse | working tree (to be committed by you) |
| Single-spike test: arrival at k − 1 before the fix, exactly k after, at Δt = 1 and 0.1 ms | Run Nov 17 on your Mac; script `scratchpad/ringtest/check_ring_buffer.py` (not in repo) |
| Brunel rerun after the fix: 110.82 / 101.13 / 80.71 spikes per neuron (+1.5%, +2.4%, −2.5% vs Brian2) | Same run, 3 seeds per p, same seeds as `run_all.py` |
| The Brian2-comparison run was almost certainly on your Mac | Rerunning eb02eb9 with the same seeds reproduced every logged count exactly and the ~33 s runtime |
| The OpenMP version wrote to (read index + d), so its delays were correct | `git show 084dd31:simulations/clock_driven_openmp/backend.cpp` line 174 |
| Event struct 16 bytes; std::priority_queue with greater<> | `simulations/event_driven/backend.cpp` lines 17–30 (4+4+4+1, padded to 16) |
| Event-driven: per-neuron dt, std::exp per event; current held constant over the interval when advancing V | `event_driven/backend.cpp` lines 160–172 |
| Poisson inputs re-arm with exponential inter-arrival times | `event_driven/backend.cpp` lines 121–122, 155–156 |
| ~24 tensor ops per clock-driven step | My count of ops in the 63ab99a dense loop body (approximate) |
| No `.item()` / host copy in dense loop | Dense `run()` body: integer indexing only |
| Sparse loop indexes with a 0-dim device tensor (possible sync) | `clock_driven_sparse/backend.cpp` line 132. That libtorch calls `.item()` there is my understanding of `Tensor::operator[](Tensor)`, **not verified** |
| Preallocation, in-place ops, accessors, SIGALRM/alarm timeout, sig_atomic_t flag | `backend.cpp` files; `runner.py` files |
| No `_out` ops used | grep of `backend.cpp` files |
| blake2b seeds at HEAD; Python `hash()` seeds at 63ab99a (salted per process) | `shared/utils.py` `get_deterministic_seed`; `git show 63ab99a:simulations/run_all.py` lines 48–60 |
| Timing: only `run()`, `cuda.synchronize()` before and after, `time.time()` | `shared/monitoring.py`, `runner.py` files |
| pybind11 ≈ MATLAB MEX gateway; libtorch = PyTorch's C++ library | General knowledge (external) |

## Results (slides 3, 9, 15–21)

All values are from 04-30 unless marked otherwise. Dense CPU uses **only** file lines 2–37 (see mismatch 1 below).

| Claim | Value | Source |
|---|---|---|
| Winner map | per cell | Derived: `derived.winner` |
| Event-driven lead at N ≤ 100 | 8.7× (N=100, p=1) to 1,592× (N=10, p=0.05) | Derived |
| N = 10³: event-driven wins p ≤ 0.05; dense GPU from p = 0.1 | 0.228 vs 0.371 s; 0.488 vs 0.372 s | 04-30 |
| N = 10⁴: sparse GPU wins p ≤ 0.5, dense GPU at 0.75 and 1 | 1.503 vs 1.923; 1.972 vs 1.924 s | 04-30 |
| Dense GPU N = 10⁴ flat | 1.9234–1.9237 s | 04-30 |
| Dense GPU floor N ≤ 10³ | 0.358–0.401 s (N=10 p=0.01 outlier 0.627 s) | 04-30 |
| 0.37 ms/step, ~15 µs/op | Derived: 0.37 s ÷ 1000 steps ÷ ~24 ops | — |
| 1.2 GB weights per step at N = 10⁴ | Analytic: 3 × 10⁸ × 4 B | — |
| 624 GB/s effective | Derived: 1.2e12 B ÷ 1.923 s | — |
| ~672 GB/s RTX 4070 Ti Super bandwidth | **External** (vendor spec) | — |
| Sparse CPU N = 10⁴ | 2.10 s (p=0.01) … 12.5 (0.1) … 173.2 (1) | 04-30 |
| CPU sparse/dense crossover N = 10⁴ | 75.7 vs 92.0 (p=0.4); 98.4 vs 87.4 (p=0.5) | 04-30 |
| Event-driven N = 10³ | 0.0454 s (p=0.01) → 11.62 s (p=1); ×256 | 04-30, derived ratio |
| Event-driven N = 10⁴ | 8.04 s (p=0.01), 66.8 s (p=0.05); p ≥ 0.1 absent | 04-30 |
| 142× | 88.70 ÷ 0.6228 (dense CPU vs sparse GPU, N=10⁴, p=0.01) | Derived |
| 3.4× | 2.099 ÷ 0.6228 | Derived |
| Per-delivery cost 122–303 ns | runtime ÷ (N · spikes/neuron · ⌊N²p⌋/N + N·50) | Derived: `derived.event_ns_per_event` |
| Firing 23 → 45 Hz (event, N=10³) | 22.85 → 45.0 spikes/neuron | 04-30 |
| Dense CPU N = 100 slower than N = 10³ | 3.6–4.5 s vs ~1.0 s | 04-30 (unexplained) |
| Clock-driven backends agree within 1.3% | max (max − min)/mean over 18 points, N ≥ 10³ | Derived: `derived.clock_spread_max_pct` = 1.32 |
| Event-driven 4.3–7.4% lower | 11 points, N ≥ 10³ | Derived: `derived.event_gap_pct` |
| Brian2 vs dense: +1.6%, +2.4%, −2.7% | means of 3: 110.84/109.13, 101.12/98.72, 80.55/82.75 | HEAD csv; `derived.brian2_diff_pct` |
| Brian2 1.2–1.6 s vs dense 33.4–33.7 s (mean per p) | individual runs 0.55–1.96 s and 32.9–33.8 s | HEAD csv |
| Run-to-run ranges in the Brian2 table | min–max of 3 runs | HEAD csv |
| Bytes per CSR edge ≈ 12 (float32 + int64 index) | Analytic: torch CSR default index dtype is int64 | `shared/clock_driven.py` (COO → CSR) |
| Memory table (backup 1) | Analytic | `shared/clock_driven.py`, `shared/graph_creation.py` |

## What didn't work (slide 22, backups 5–6)

| Claim | Source |
|---|---|
| torch.compile first call 1419, 1594, 1537, 1689 s (N = 10…10⁴) | `logs/04-16/output.txt` |
| Warm compiled vs eager: 0.0069 vs 0.243 s (N=10); 197 vs 97.6 s (N=10⁴); 1.39× slower at N=10³ | `logs/04-16/output.txt` (mean of runs 2–5 vs mean of 5) |
| GPU compiled: 3516 s first call at N = 10; N = 100 RecursionError after 1111 s; CUDA Graphs skipped due to mutated inputs | `logs/04-16/output.txt` lines 245–300 |
| 04-16 ran on Colab with the Python-loop version | `/content/snn-scaling` paths in `logs/04-16/output.txt`; commit order (C++ move is 9583efd, after 04-16) |
| OpenMP: push model, `#pragma omp atomic`, `.item<bool>()` per neuron, Tensor indexing per synapse | `git show 084dd31:simulations/clock_driven_openmp/backend.cpp` lines 149–193 |
| No `-fopenmp` in setup.py (compile or link) | `git show 084dd31:setup.py` (`_EXTRA_COMPILE_ARGS = ["-O3", "-march=native"]`) |
| OpenMP 70.7 s vs event-driven 68.5 s at N=10³, p=1, same run | `git show 084dd31:results.csv` |
| Added May 1, deleted May 5 | commits 084dd31, 4494bfc |

## Limitations (slide 23)

| Claim | Source |
|---|---|
| Only wall-clock runtime measured | `shared/monitoring.py` (time.time only); no memory/energy columns in any CSV |
| Energy prototyped on a MacBook in March | commits 027ee12, 2d65056 (`zeus_apple_silicon`); removed by 97381ee |
| No test suite; one ad-hoc single-spike test exists | no test files in repo; test script in scratchpad |
| SpiNNaker: virtual board only | `spynnaker.cfg` `virtual_board = True`; README status; `logs/04-05/output.txt` ("Virtual machine generator") |
| Sparse/event runners stale at HEAD | `simulations/clock_driven_sparse/runner.py`, `simulations/event_driven/runner.py` reference fields absent from `shared/simulation_config.py` |
