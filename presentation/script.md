# Speaker script: spiking-network execution strategies on CPU and GPU

Slides 1–25 are planned at **43.5 min**; with transitions that comes to about 45 min, then 15 min of questions.

**If you run long**, cut these slides (marked [CUT IF LONG]): 5 (Two-pole model, 1.5 min), 13 (Event-driven loop, 1.5 min), 17 (Why event-driven differs, 1 min), 21 (Sparse GPU, 1.5 min). Together they save about 5.5 min.

The same text is in each slide's speaker notes. "If asked" lines are prompts for likely questions.

## Slide 1: Title (0.5 min, elapsed 0.5)

- Good morning. I'm Cannon; I build software that has to be correct and fast — most recently test and data tooling at CACI, before that an M.S. in CS at UW–Madison with an HPC focus.
- This talk is a personal project: the same simulation run three different ways, measured on CPU and GPU.
- I'll keep the neuroscience to a minimum — the core of it is a recursive filter, a sparse matrix, and a priority queue.

## Slide 2: Question (2 min, elapsed 2.5)

- Same network, same parameters; only the execution strategy and device change.
- Clock-driven: advance every neuron every time step — like sampling every channel at a fixed rate. Dense stores every possible connection; sparse stores only real ones (CSR).
- Event-driven: only do work when a spike arrives, scheduled in a priority queue.
- Two questions: where are the crossovers, and can I predict them from the primitive — dense GEMV, sparse SpMV, heap operations.
- If asked "why does anyone care": choosing a simulator today is mostly folklore; I wanted a cost model I could check against measurements.

## Slide 3: Preview (1.5 min, elapsed 4)

- Here's the answer before the details. Each cell is the fastest backend for one network size and connection probability.
- Small networks: event-driven wins by orders of magnitude, because clock-driven pays a fixed cost every step even when nothing happens.
- Large networks: the GPU wins everywhere; sparse up to p = 0.5, dense above that.
- The cost models get the shapes right; the exact crossover depends on hardware constants I'll show later.
- Caveat to say out loud: one run per cell, single machine. These are first measurements, not a survey.
- If asked "what's the blank cell": at N = 10, p = 0.01 nothing spikes and there's no event-driven row in the log.

## Slide 4: LIF as a one-pole filter (2 min, elapsed 6)

- If you remember one thing about the neuron: it's a one-pole IIR low-pass filter with a comparator on the output.
- Input is a sum of weighted spikes; the state decays by alpha each step, alpha = e^(−dt/tau). That's the exact discrete-time solution of the RC circuit, not an Euler approximation.
- When the voltage crosses threshold it emits a spike, resets, and holds for a refractory period. That's the only nonlinearity.
- Tau = 10 ms gives a cutoff around 16 Hz — it smooths the input spike train.
- If asked "why exact instead of Euler": with a constant input between steps the exponential is the closed form; it costs the same and removes a stability limit on dt.

## Slide 5: Two-pole model [CUT IF LONG] (1.5 min, elapsed 7.5)

- The benchmark sweep used a slightly richer neuron: the input first goes through a synaptic-current filter, then the membrane filter. Two cascaded one-pole low-pass filters.
- Poles at 0.82 and 0.90 for a 1 ms step. The (1 − alpha) term gives unity DC gain.
- I want to be upfront: the big April sweep used this model, and later I moved the dense backend to the simpler one-pole model to match Brian2 exactly. I'll be clear about which results use which.
- If asked "does the model change the performance results": the per-step work is the same element-wise ops plus one extra multiply-add per neuron; the dominant costs — mat-vec, SpMV, heap — don't change.

## Slide 6: Network (2 min, elapsed 9.5)

- The network is a random directed graph: every possible edge exists independently with probability p, and each edge gets its own delay, uniform between 2 and 4 ms.
- External drive is Poisson spike input — think of it as a noise floor that keeps neurons near threshold.
- Two configurations, and I'm keeping them separate on purpose. Left: the big benchmark sweep, all-excitatory, 1 ms step. Right: a Brunel-2000-style excitatory/inhibitory network I used to validate against Brian2, at 0.1 ms.
- The 20 ms refractory period in the sweep caps firing at 50 Hz — at large N neurons run near that cap (about 48 Hz). That makes spike counts nearly constant across p, which matters for reading the event-driven results.
- If asked "why exact edge counts": it removes edge-count variance between backends so runtime differences aren't graph noise.

## Slide 7: Clock vs event (1.5 min, elapsed 11)

- Clock-driven: every neuron, every step. Cost scales with steps times neurons, whether or not anything happens. Fixed-rate sampling.
- Event-driven: keep a queue of future spike arrivals; pop the earliest, update only that neuron. Because each neuron is updated at irregular times, the decay factor is computed per event with its own dt.
- The trade: clock-driven has regular, vectorizable work — great for GPUs. Event-driven skips idle time but does irregular, pointer-chasing work that's hard to parallelize.
- If asked about accuracy: event-driven has no time grid, so spike times are not rounded to dt; clock-driven rounds delays and spike times to the step.

## Slide 8: Primitives (2 min, elapsed 13)

- Strip away the neuroscience and each backend is one primitive.
- Dense: one N×N weight matrix per delay bucket, multiplied by the 0/1 spike vector every step — a batched GEMV. Same work at p = 0.01 as at p = 1.
- Sparse: the same, but each bucket is a CSR matrix, so work scales with the number of stored edges. Note it still multiplies the full spike vector — it exploits sparse connectivity, not sparse activity.
- Event-driven: a binary heap. Each spike pushes N·p future arrivals; each arrival is a pop plus a neuron update.
- If asked "why not skip non-spiking columns in the dense case": you could gather the spiking columns first; at 25–48 Hz with 1 ms steps only 2–5% of neurons spike per step, so that's a real optimization I didn't try.

## Slide 9: Cost models (2.5 min, elapsed 15.5)

- Dense: steps × buckets × N². No p, no firing rate. Bytes per step are 4·B·N² — 1.2 GB per step at N = 10,000 with 3 buckets.
- Sparse: proportional to stored edges, N²p. About 12 bytes per edge: a float value plus a 64-bit column index.
- Event-driven: number of spikes times out-degree times log of the queue length.
- Clock-driven also has a fixed floor: steps × ops per step × per-op overhead. That floor decides small-N behavior.
- Three checks from the log: dense GPU dead flat in p; sparse CPU roughly linear in p at large N; event-driven grows faster than p because the firing rate also rose.
- If asked "where's log Q": the per-event cost rises from about 120 to 300 ns as N grows — consistent with a deeper heap and more cache misses, but I haven't separated the two.

## Slide 10: Architecture (2 min, elapsed 17.5)

- Python does what Python is good at: the sweep, config validation, building the graph, allocating every tensor up front.
- The time loop is C++. Each backend is a small C++ class with the same constructor shape and a run() that returns spike counts and a timed-out flag — one interface, so the comparison is fair.
- Tensors cross the pybind11 boundary by reference; no copies. If you've written MEX files, pybind11 plays the same role as the mexFunction gateway.
- Timing wraps only run(), with a CUDA synchronize on both sides, so setup cost isn't counted.
- If asked "why move to C++": in the Python-loop version, interpreter overhead per step was a measurable part of runtime; moving the loop removed that confound across backends.

## Slide 11: Per-step data flow (2 min, elapsed 19.5)

- Five stages per step. Draw the Poisson drive, run the two filters, threshold, propagate through the weights into the ring buffer, then reset and record.
- Stage 4 is the matrix work; everything else is element-wise over N neurons.
- I counted roughly 24 tensor ops per step. On a GPU each is a kernel launch — keep that number in mind for the dense-GPU results.
- The dense loop never copies to the host or calls .item() — spike counters live on the device and are read once at the end. That's the "keep data on the device" rule.
- If asked "what about the sparse loop": it indexes the ring buffer with a 0-dim device tensor; I believe libtorch turns that into an .item() call, meaning a device sync three times per step. That's a hypothesis I'd check with Nsight Systems.

## Slide 12: Delay buckets and ring buffer (key slide) (2.5 min, elapsed 22)

- This is the key data structure. Delays are continuous, but a clock-driven sim can only deliver on step boundaries, so I quantize each delay to whole steps and group synapses into buckets — one weight matrix per bucket.
- Each step: multiply each bucket's matrix by the spike vector and add the result into a ring buffer slot that many steps in the future. Read the current slot, zero it, advance. No queue, no sorting — just a circular buffer of future inputs, like a tapped delay line.
- Two honest findings from re-reading the code for this talk. One: with delays uniform on [2, 4) ms, the third bucket is always empty, so a third of the dense mat-vec work is multiplying zeros. Two: the read pointer advanced before the read, so contributions arrived one step early — effective delays of 1–2 ms instead of 2–3 in the April sweep. Dense and sparse both had it; the old OpenMP version did not.
- I've fixed the second one: the read slot is now simply t mod B. A one-spike test — one synapse, known delay — failed before the fix and now arrives on exactly step k at both 1 ms and 0.1 ms.
- Neither changes the scaling story. The lesson is that a unit test on a single known spike would have caught it on day one.
- If asked "how big can the buffer get": slots = max delay / dt + 1 — 41 slots at dt = 0.1 ms. It's O(N) per slot, tiny next to the weights.

## Slide 13: Event-driven loop [CUT IF LONG] (1.5 min, elapsed 23.5)

- The whole event-driven engine is a std::priority_queue of 16-byte Event structs, ordered by time.
- Pop the earliest, advance just that neuron using its own elapsed time, check threshold. A spike pushes one event per outgoing edge at t plus that edge's delay.
- Poisson inputs are events too: each one schedules the next arrival with an exponential inter-arrival time — that's an exact Poisson process, not a per-step coin flip.
- Dividing runtime by spike deliveries gives about 120–300 ns per delivery, rising with N — a bigger heap and more cache misses.
- If asked "why float time": float32 has about 60 ns resolution at t = 1 s, fine for 1 s runs; for longer runs I'd use double or integer ticks.

## Slide 14: Engineering details (1.5 min, elapsed 25)

- Five habits. Allocate everything before the loop and update in place. Pass tensors by handle, never copy. Seed deterministically. Bound every run. Time only the part you mean to time.
- The timeout is a classic signal pattern: SIGALRM sets a sig_atomic_t flag and the loop checks it — no threads, no exceptions across the C++/Python boundary.
- Honest note: the April sweep's seeds came from Python's built-in hash, which is randomized per process, so those exact graphs can't be regenerated. I switched to a blake2b hash afterward.
- If asked "what happens to timed-out runs": they're dropped, which is why some event-driven points at N = 10,000 are missing. Better practice is to record them as censored at 300 s; that's on the list.

## Slide 15: Cross-backend validation (1.5 min, elapsed 26.5)

- First check: do my own backends agree? Same parameters, different code paths — dense versus sparse, CPU versus GPU.
- The four clock-driven backends agree within 1.3% on mean spike count at every point with N ≥ 1,000. They use different seeds, so that includes graph-to-graph variation.
- Event-driven is consistently lower, 4 to 7%. Consistent and one-sided means a systematic modeling difference, not noise — next two slides.
- Caveat: mean rate is a coarse statistic. Next steps are rate distributions and a single-neuron test against the closed form.
- If asked "why not compare spike trains": different seeds and float rounding make spike-by-spike comparison meaningless; statistics are the right level.

## Slide 16: Brian2 reference (1.5 min, elapsed 28)

- Second check: an independent simulator. Brian2 is a widely used Python SNN simulator. I matched its model and parameters to the classic Brunel 2000 network.
- Mean rates agree within 1.6 to 2.7%. At p = 0.3 the difference is inside the spread across seeds; at the lower p values it's small but outside it, so there's still a systematic piece I haven't pinned down.
- Brian2 was about 20 times faster on this setup. My dense backend multiplies 21 bucket matrices every step at dt = 0.1 ms; Brian2 propagates only actual spikes. That's the same primitive story as the rest of the talk.
- After I fixed the ring-buffer offset I reran these exact seeds: rates moved by less than 0.2% and the gap stayed at +1.5, +2.4 and −2.5%. So that bug isn't the explanation.
- If asked about the gap: remaining candidates are exact edge count versus Bernoulli connections, where the inhibitory neurons sit, and floor-versus-round delay quantization.

## Slide 17: Why event-driven differs [CUT IF LONG] (1 min, elapsed 29)

- Why is event-driven 4 to 7% lower? Event-driven delivers a 3.6 ms delay at 3.6 ms. The clock-driven code floors it to 3 ms, and in the April sweep an indexing offset delivered it at 2. That offset is now fixed.
- Two other differences: event-driven treats the synaptic current as constant over each interval when it updates the voltage, and the clock-driven input is a coin flip per step rather than a true Poisson process.
- I haven't separated these yet. The clean experiment is a dt sweep — discretization error should shrink with dt. The offset fix is done and tested; at 0.1 ms it barely moved rates, which is expected because the shift there is only 0.1 ms.
- If asked "which do you think dominates": at 1 ms the offset was a full step out of a 2.5 ms delay, so it could matter there; I won't know until the 1 ms sweep is rerun.

## Slide 18: MAIN FIGURE (2.5 min, elapsed 31.5)

- This is the main result. Runtime against connection probability, both axes log, N = 1,000 left and 10,000 right. One simulated second, 1 ms steps.
- Left: event-driven (red) starts lowest and rises with p — at p = 0.01 it's 45 ms. Dense GPU sits flat at about 0.37 s and takes over from p = 0.1.
- Right: everything moves up. Dense CPU is about 85–94 s regardless of p. Sparse CPU rises from 2 s to 173 s. Sparse GPU is the fastest up to p = 0.5; dense GPU is flat at 1.92 s and wins above that.
- Event-driven at N = 10,000 is already 8 s at p = 0.01 and 67 s at 0.05; larger p weren't recorded.
- Say clearly: one run per point. I'd put confidence intervals on this before publishing anything.
- If asked "why does dense CPU at N = 10⁴ wander between 70 and 94 s": single runs on a shared-memory Xeon — I'd expect run-to-run variance from memory bandwidth contention; repeats would show it.

## Slide 19: Event-driven crossover (1.5 min, elapsed 33)

- Same data sliced the other way: runtime versus N at p = 0.05.
- At small N, event-driven is two to three orders of magnitude ahead. Clock-driven backends sit on a floor — a fixed cost per step for launching ops — no matter how little there is to do.
- At N = 1,000 it still wins at low p. At N = 10,000 it never wins: the number of spike deliveries grows like N squared times p times the firing rate, and each costs 120–300 ns on one core, while the GPU stays near its floor.
- There's an oddity: dense CPU is slower at N = 100 than at 1,000. I haven't investigated; my guess is thread-pool overhead on tiny matrices. I'd rather show it than hide it.
- If asked "would a parallel event-driven backend change this": yes, potentially — see the backup slide on lookahead parallelism.

## Slide 20: Dense GPU (2 min, elapsed 35)

- Dense GPU is the most instructive curve because it's flat — p doesn't matter to a dense mat-vec.
- For N up to 1,000 it takes 0.36 to 0.40 seconds regardless of size: about 0.37 ms per step. With about 24 ops per step, that's roughly 15 microseconds each. The GPU is mostly waiting on launches and dispatch, not computing.
- At N = 10,000 it's 1.92 s. Each step reads 1.2 GB of weights; divide and you get about 624 GB/s, close to the card's published ~672 GB/s. So at that size it's memory-bandwidth bound — the right kind of bound for a GEMV.
- Two cheap wins: CUDA Graphs or fusion for the floor, and dropping the always-empty bucket for a third of the bandwidth. Neither is measured yet.
- If asked "is 15 µs per launch realistic": raw launch latency is a few microseconds; libtorch dispatch, the caching allocator and small temporaries add the rest. I'd confirm with Nsight Systems rather than assert it.

## Slide 21: Sparse GPU [CUT IF LONG] (1.5 min, elapsed 36.5)

- At N = 10,000 the sparse GPU backend is the fastest thing I built: 0.62 seconds at p = 0.01, 142 times faster than dense CPU and 3.4 times faster than sparse CPU.
- The byte model says dense and CSR read the same number of bytes only at p = 1, so naively sparse should win everywhere. It loses above p ≈ 0.6 because CSR access is irregular and the sparse loop has a higher fixed floor.
- On the CPU the crossover is lower, between 0.4 and 0.5.
- My hypothesis for the higher sparse floor is an implicit device sync from indexing with a device tensor. That's testable in an afternoon with Nsight.
- If asked "why int64 indices": that's PyTorch's default for CSR. Switching to int32 would cut index bytes in half; another cheap experiment.

## Slide 22: What didn't work (3 min, elapsed 39.5)

- Two things that didn't work, and I think the reasons matter more than the results.
- torch.compile: compiling took 24 to 28 minutes per network size. It helped tiny networks — 35 times faster at N = 10 because fusion removes per-op overhead — but at N = 10,000 it was twice as slow as eager. On the GPU it couldn't use CUDA Graphs because I mutate a dynamically indexed slice of the ring buffer, and at N = 100 the compiler crashed. I dropped it and moved the loop to C++.
- OpenMP: I wrote a push-style backend that scatters each spike's targets into the ring buffer with atomics. It was as slow as the single-threaded event-driven code, so I deleted it. Re-reading it for this talk: setup.py never passed -fopenmp, so it was never parallel. And the inner loop did a libtorch tensor index per synapse — microseconds each. Adding the flag alone wouldn't fix that; it needs raw pointers.
- Lesson I took: verify the build actually did what you think, and profile before parallelizing.
- If asked "how would you do it now": see backup — pull model over target neurons, no atomics.

## Slide 23: Limitations and next steps (2 min, elapsed 41.5)

- I want to be precise about scope. What I measured is wall-clock runtime of the simulation loop. Memory I can compute exactly but didn't measure. Energy was planned — I prototyped it on a MacBook early on, but I have no energy numbers from this hardware.
- Line to use if someone points at the résumé: "My résumé lists latency, throughput, memory and energy as the project's axes. What I've measured so far is runtime — throughput falls out of that as events per second. Memory is analytic, and energy instrumentation is the next step."
- No test suite yet, which is how an indexing bug survived. I've fixed the ring-buffer offset this week with a single-spike test; the empty bucket and the stale sparse and event runners are next.
- SpiNNaker: the runner exists but has only run against a virtual board, never real hardware.
- Next steps in priority order: tests and the bug fixes, repeats with confidence intervals, the dt sweep, then memory and energy, then performance work — CUDA Graphs, a calendar queue, a correct parallel CPU backend.

## Slide 24: Takeaways (2 min, elapsed 43.5)

- Five things I'd bring to array-processing work.
- A ring buffer of future contributions is the same structure as a circular buffer of sensor blocks or a delay line.
- Quantizing delays to the sample grid is the integer-versus-fractional-delay problem in time-domain beamforming — and my off-by-one is the kind of bug that only a test on a known delay catches.
- Small problems drowned in per-step overhead: batch work into blocks big enough to amortize launches, and keep data on the device.
- Match the primitive to the data's density.
- And validate against a reference — Brian2 here, a MATLAB reference in a sonar chain — before trusting any speedup.
- Close: measure, model, validate, and be explicit about what you didn't measure. Thank you.

## Slide 25: Questions (0 min, elapsed 43.5)

- Leave this up during Q&A. Backup slides follow: memory math, ring-buffer sizing, full runtime tables, OpenMP redesign, torch.compile numbers, Brian2 parameter matching, parallel event-driven, floating-point reproducibility.

---

## Backup slides (Q&A only)

### Slide 26: memory

- Dense memory is 4·B·N², and B grows as dt shrinks: 3 buckets at 1 ms, 21 at 0.1 ms. At N = 10,000 and 0.1 ms that's 8.4 GB just for weights.
- Construction is worse than steady state: building the graph allocates N² int64 permutations and masks.
- None of this was measured — peak-allocation measurement is on the next-steps list.

### Slide 27: ring buffer

- Walk through the indexing if asked. The read index advances before the read, the write uses t plus offset, so arrival is one step early.
- I found this re-reading the code for the talk and fixed it: the read slot is now t mod B. The single-spike test failed before (arrival at k − 1) and passes after.

### Slide 28: runtime table

- Every number behind the main figure. Fastest per column is tinted.
- Be ready for: why is there a dash — event-driven at N = 10,000, p ≥ 0.1 wasn't recorded.
- Data-hygiene note if asked: the CSV has a second block labelled dense CPU that can't be dense CPU; I excluded it. The README plot averages it in, so that plot's dense-CPU curve is wrong.

### Slide 29: small-N table

- Small networks: clock-driven runtimes are floors — 0.1 to 4 s — while event-driven is microseconds to milliseconds.

### Slide 30: OpenMP

- As written, without -fopenmp the pragmas are ignored. Even with the flag, omp atomic on a Tensor expression isn't a scalar update, and every inner iteration goes through libtorch indexing.
- Redesign: raw pointers, pull model over targets with a transposed graph so no atomics, or per-thread buffers with a reduction.

### Slide 31: torch.compile

- Compile helps tiny problems, hurts big ones, and the compile itself costs tens of minutes. On GPU it couldn't use CUDA Graphs because of the in-place zero of a dynamically indexed slice.

### Slide 32: Brian2 matching

- Differences that remain: exact versus Bernoulli edge counts, inhibitory neuron placement, and how each rounds delays to the grid. Any of these could explain a 2% rate difference.

### Slide 33: parallel event-driven

- The classic answer is conservative synchronization with lookahead equal to the minimum delay. With 2 ms minimum delay and ~120–300 ns per event, each window has plenty of work per thread at N = 10⁴.

### Slide 34: FP reproducibility

- Chaotic dynamics plus non-associative float sums means bitwise reproducibility is the wrong target. Statistics are.

### Slide 35: provenance

- If someone asks "could I rerun this": the Apr 30 sweep corresponds to commit 63ab99a, which still has the ring-buffer offset; HEAD has moved the dense backend to the Brunel model and the other runners need updating.
