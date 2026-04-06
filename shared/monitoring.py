import time


class MonitoringWindow:
    """
    Context manager for measuring runtime and energy/memory consumption of a code block

    TODO: support intel/nvidia/spinnaker platforms and use CUDA synchronize
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {elapsed_time} seconds.")
