# Simulations

These implementations are designed for benchmarking CPU vs GPU performance for simulating a spiking neural network with dense vs sparse operations under controlled scaling conditions. Activity is driven by external Poisson input and recurrent connectivity. The function uses current-based LIF neurons with no conductances and no refractory periods. The model is minimal to isolate scaling behavior from biological complexity. Recording and weight updates are not done.

Updates to membrane voltages take this form:

V[t+1] = V[t] * exp(-dt/tau_m) + I_syn * (1 - exp(-dt/tau_m))

Synaptic current decays exponentially and is incremented by recurrent spikes via weights @ spikes and by Poisson input.

The `clock_driven` directory contains simulations that run according to fixed timesteps. The `event_driven` directory contains simulations that run according to events.