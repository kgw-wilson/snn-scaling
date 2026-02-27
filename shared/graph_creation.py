from __future__ import annotations

import math
import torch
from shared.simulation_config import ERGraphConfig


def create_weighted_er_graph(config: ERGraphConfig) -> torch.Tensor:
    """
    Create a weighted Erdos-Renyi weighted connectivity matrix

    This function generates a dense N x N weight matrix W suitable for
    benchmarking SNN simulation performance across CPU and GPU.

    Design decisions:

    1. Dense tensor representation
       Even though connectivity is probabilistic (p), the result is stored
       as a dense tensor to enable fast BLAS-backed matrix multiplication and
       to allow fair CPU vs GPU comparison on dense graphs.

    2. Erdos-Renyi connectivity
       Each directed edge is included independently with probability p.
       No structural bias (i.e. no small-world or scale-free structure).
       This isolates scaling behavior with respect to N and p.

    3. Weight initialization
       Raw weights are drawn from N(0, 1) before masking.
       This keeps statistics controlled and reproducible.
       After masking, weights are scaled by:

           g / (sqrt(N) * p)

       Rationale:
       - Expected in-degree ≈ pN
       - Variance of summed input should remain O(1) as N grows
       - sqrt(N) scaling keeps dynamics size-stable
       - Division by p compensates for sparsity

       This prevents trivial blow-up or vanishing activity as N varies,
       which is critical when measuring scaling laws.

    4. Zero row mean (masked)
       For each neuron i, subtract the mean of its nonzero outgoing
       weights. This enforces approximate balance and reduces drift
       in aggregate input currents. The subtraction is applied only
       to existing edges via a mask.

    Parameters:
        config - ERGraphConfig
            Instance of dataclass containing validated graph properties

    Returns:
        torch.Tensor
            Dense weight matrix of shape (N, N) on specified device.
    """

    # Use seed for reproducibility
    torch.manual_seed(config.seed)

    # Sample full dense Gaussian matrix
    W = torch.randn((config.N, config.N), dtype=config.dtype, device=config.device)

    # Create connectivity mask
    mask = torch.rand((config.N, config.N), device=config.device) < config.p

    # Apply sparsity by zeroing out non-existant connections
    W = W * mask

    # Apply balanced scaling
    W = W * (config.g / (math.sqrt(config.N) * config.p))

    # Zero masked row mean
    row_sum = W.sum(dim=1, keepdim=True)
    row_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    row_mean = row_sum / row_count
    W = W - row_mean * mask

    return W
