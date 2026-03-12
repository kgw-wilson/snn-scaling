# Event-driven simulations

Calculations in the event-driven simulations are similar to the clock-driven ones, with these changes:

- Spikes are scheduled within a min-heap priority queue rather than in a tensor. This datastructure can be quickly extended when new spikes are generated at a future time and can be easily popped to get the next scheduled spike.

- The loop condition has changed to check the length of the queue holding scheduled spike events. Poisson spikes are generated for the whole simulation, so the simulation should run for the entire simulation duration.

- Update times are variable. In the clock-driven simulation, updates occured for every neuron at every timestep, so the voltage/current decay values were constant and thus pre-computed. In the event-driven simulations, different neurons update at different times, so per-neuron update times and decay values are calculated each time in the loop.
