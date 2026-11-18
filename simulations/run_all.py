import subprocess
import sys

from shared.reporting import RESULTS_PATH
from shared.utils import get_available_devices
from simulations.run_one import TIMED_OUT_EXIT_CODE

_CONNECTION_PROBS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
_NUM_NEURONS = [10, 100, 1000, 10000]
_NUM_REPEATS = 3
_NUM_THREADS = 8

_DEVICE_TO_SIMULATIONS = {
    "cpu": [
        "clock_driven_dense",
        "clock_driven_sparse",
        "clock_driven_openmp",
        "event_driven_cpu",
    ],
    "gpu": [
        "clock_driven_dense",
        "clock_driven_sparse",
    ],
    "neuromorphic": [],
}


if __name__ == "__main__":

    if RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} exists, move it before starting a new run.")

    # (simulation, device, num_neurons) series that timed out; runtime grows with
    # connection probability, so larger probabilities for that series are skipped
    timed_out = set()

    for device in get_available_devices():

        for simulation in _DEVICE_TO_SIMULATIONS[device]:

            for num_neurons in _NUM_NEURONS:

                for connection_prob in _CONNECTION_PROBS:

                    for repeat_i in range(_NUM_REPEATS):

                        if (simulation, device, num_neurons) in timed_out:
                            continue

                        print("===================", flush=True)
                        print(
                            f"{simulation}, {device=}, {num_neurons=}, {connection_prob=}",
                            flush=True,
                        )

                        result = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "simulations.run_one",
                                simulation,
                                device,
                                str(num_neurons),
                                str(connection_prob),
                                str(repeat_i),
                                str(_NUM_THREADS),
                            ]
                        )
                        if result.returncode == TIMED_OUT_EXIT_CODE:
                            timed_out.add((simulation, device, num_neurons))
                            print(
                                f"{simulation} timed out, skipping larger connection probabilities",
                                flush=True,
                            )
                        elif result.returncode != 0:
                            print(
                                f"{simulation} exited with code {result.returncode}",
                                flush=True,
                            )

    print("All simulations completed.")
