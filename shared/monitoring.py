import time
from zeus_apple_silicon import AppleEnergyMonitor


class MonitoringWindow:
    """
    Context manager for measuring runtime and energy/memory consumption of a code block

    Currently only supports Apple devices but will be extended to support other platforms.
    
    TODO: support other platforms and use CUDA synchronize
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.monitor = AppleEnergyMonitor()

    def __enter__(self):
        self.monitor.begin_window(self.name)
        self.start_time = time.time()
        return self.monitor

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {elapsed_time} seconds.")
        mes = self.monitor.end_window(self.name)
        print(mes)
