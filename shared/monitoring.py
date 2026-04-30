import time
import torch


class MonitoringWindow:
    """
    Context manager for measuring runtime and energy/memory consumption of a code block

    TODO: support intel/nvidia/spinnaker platforms and use CUDA synchronize
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        try:
            torch.cuda.synchronize()
        except:
            pass
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            torch.cuda.synchronize()
        except:
            pass
        self.elapsed_time = time.time() - self.start_time
