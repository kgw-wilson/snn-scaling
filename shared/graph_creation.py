from __future__ import annotations

import math
import torch
from shared.simulation_config import ERGraphConfig


def create_er_dense(config: ERGraphConfig) -> torch.Tensor:
    """
    Create a dense weighted Erdos-Renyi connectivity matrix

    This function generates a dense N x N weight matrix W suitable for
    benchmarking SNN simulation performance on CPU and GPU.

    Each directed edge is included independently with probability p and
    no structural bias (i.e. no small-world or scale-free structure).
    This isolates scaling behavior with respect to N and p.

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

    For each neuron i, the  mean of its nonzero outgoing eights is
    subtracted. This enforces approximate balance and reduces drift
    in aggregate input currents. The subtraction is applied only
    to existing edges via a mask.

    Returns:
        torch.Tensor
            Dense weight matrix of shape N x N on specified device
    """

    torch.manual_seed(config.seed)

    weights = torch.randn(
        (config.num_neurons, config.num_neurons),
        dtype=config.dtype,
        device=config.device,
    )

    connectivity_mask = (
        torch.rand((config.num_neurons, config.num_neurons), device=config.device)
        < config.connection_prob
    )

    weights = weights * connectivity_mask

    weights = weights * (
        config.global_coupling_strength
        / (math.sqrt(config.num_neurons) * config.connection_prob)
    )

    outgoing_weight_sums = weights.sum(dim=1, keepdim=True)
    num_outgoing_connections = connectivity_mask.sum(dim=1, keepdim=True).clamp(min=1)
    outgoing_weight_means = outgoing_weight_sums / num_outgoing_connections

    weights = weights - outgoing_weight_means * connectivity_mask

    return weights
