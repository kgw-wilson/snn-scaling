import time

import psutil
import torch

_process = psutil.Process()

run_rss_bytes = {"start": None, "end": None}


class MonitoringWindow:
    """
    Context manager for measuring the runtime of a code block

    Also records the process's resident memory just before and just after the
    block (outside the timed region) in run_rss_bytes.
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        run_rss_bytes["start"] = _process.memory_info().rss
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time
        run_rss_bytes["end"] = _process.memory_info().rss
