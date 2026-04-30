# Logs

Unstructured output containing logs and analysis for simulation runs. Future results will be more organized, but this folder now contains initial testing.

- `04-05` : All simulations except neuromorphic running with equivalent statistics for N=1000 and p=0.5

- `04-11`: First parameter sweep, including some basic plotting. All simulations except neuromorphic (and event_driven_cpu N=10000 p=1 since Colab cancelled runtime) running with:
    
    _CONNECTION_PROBS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
    _NUM_NEURONS = [10, 100, 1000, 10000]
    _NUM_REPEATS = 1

- `04-16`: Explores eager execution vs pytorch.compile for dense cpu simulation. pytorch.compile was dropped for subsequent experiments. See README in folder for more info.