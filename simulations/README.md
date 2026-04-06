# Simulations

These implementations are designed for benchmarking CPU vs GPU vs neuromorphic performance for simulating a spiking neural network. That is with event-driven and clock-driven implementations with dense vs sparse operations under controlled scaling conditions. Activity is driven by external Poisson current input and recurrent connectivity. The function uses basic LIF neurons. The model is minimal to isolate scaling behavior from biological complexity. Weight updates are not done. State updates take the analytical form instead of using Euler integration which is an approximation.

## Clock-driven Simiulations

Main simulation loops in this folder follow this flow: update currents, update voltages, find spiking neurons, schdule current in the future, perform necessary resets, and then do reporting. This order is used for correctness and to maintain biological plausibility.

## Event-driven simulations

Calculations in the event-driven simulations are similar to the clock-driven ones, with these changes:

- Spikes are scheduled within a min-heap priority queue rather than in a tensor. This datastructure can be quickly extended when new spikes are generated at a future time and can be easily popped to get the next scheduled spike.

- The loop condition has changed to check the length of the queue holding scheduled spike events. Poisson spikes are generated for the whole simulation, so the simulation should run for the entire simulation duration.

- Update times are variable. In the clock-driven simulation, updates occured for every neuron at every timestep, so the voltage/current decay values were constant and thus pre-computed. In the event-driven simulations, different neurons update at different times, so per-neuron update times and decay values are calculated each time in the loop.

## Neuromorphic

Uses SpyNNaker's wrapper around PyNN library to define and run spiking neural network simulation. Based on https://github.com/SpiNNakerManchester/PyNNExamples/blob/8919a69b70f7ae679c4ac106d719f4a4501e78d3/balanced_random/balanced_random.py
