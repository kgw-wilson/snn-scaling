from dataclasses import dataclass
import torch


@dataclass
class ERGraphConfig:
    """Configuration for Erdos-Renyi graphs.

    All values are validated after initialization.
    """

    # Random seed for random number generation for reproducibility
    seed: int

    # Number of neurons in the graph
    N: int

    # Probability of connection between two random neurons
    p: float

    # Global coupling strength scaling factor.
    g: float

    # Which device to use when allocating memory during graph creation
    device: torch.device

    # Datatype of weight matrix (torch.Tensor) after graph creation
    dtype: torch.dtype = torch.float32

    def __post_init__(self):

        # Validate seed
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}")

        # Validate network size
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError(f"N must be a positive integer, got {self.N}")

        # Validate connection probability
        if not (0 < self.p <= 1):
            raise ValueError(f"Connection probability p must be in (0,1], got {self.p}")

        # Validate gain
        if not isinstance(self.g, (float, int)) or self.g <= 0:
            raise ValueError(f"Gain g must be positive, got {self.g}")

        # Validate device and make sure requested device is available
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, got {type(self.device)}")
        else:
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available on this system")
                if (
                    self.device.index is not None
                    and self.device.index >= torch.cuda.device_count()
                ):
                    raise ValueError(f"CUDA device {self.device.index} not found")

        # Validate dtype
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(self.dtype)}")
