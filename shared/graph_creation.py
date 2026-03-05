from __future__ import annotations

import math
import numpy as np
from scipy.sparse import csr_matrix
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
    After masking, weights are scaled.

    For each neuron i, the  mean of its nonzero outgoing eights is
    subtracted. This enforces approximate balance and reduces drift
    in aggregate input currents. The subtraction is applied only
    to existing edges via a mask.

    Returns:
        torch.Tensor
            Dense weight matrix of shape N x N on specified device
    """

    if not isinstance(config.dtype, torch.dtype):
        raise ValueError(
            f"This function only supports PyTorch datatypes, got {config.dtype=}"
        )

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
    weights = weights * _weight_scalar(config)

    outgoing_weight_sums = weights.sum(dim=1, keepdim=True)
    num_outgoing_connections = connectivity_mask.sum(dim=1, keepdim=True).clamp(min=1)
    outgoing_weight_means = outgoing_weight_sums / num_outgoing_connections

    weights = weights - outgoing_weight_means * connectivity_mask

    return weights


def create_er_sparse_gpu(config: ERGraphConfig) -> torch.Tensor:
    """
    Create a sparse weighted Erdos-Renyi connectivity matrix

    This function is used when running a simulation with cuda,
    since MPS does not support sparse matrix operations as of
    03/2026. It follows the same logic (i.e. random weight sampling
    and then scaling followed by mean subtraction) as the function above.

    The weight tensor is created as coo first because of its
    more human-friendly format and then converted to csr for
    faster computation.

    Returns:
        torch.Tensor - sparse csr tensor of shape (N, N)
    """

    if config.device.type != torch.device("cuda"):
        raise ValueError(
            f"This function only supports cuda device, got {config.device=}"
        )

    if not isinstance(config.dtype, torch.dtype):
        raise ValueError(
            f"This function only supports PyTorch datatypes, got {config.dtype=}"
        )

    N = config.num_neurons
    p = config.connection_prob

    expected_edges = int(N * N * p)

    row_indices = torch.randint(N, (expected_edges,), device=config.device)
    col_indices = torch.randint(N, (expected_edges,), device=config.device)

    values = torch.randn(expected_edges, device=config.device, dtype=config.dtype)
    values *= _weight_scalar(config)

    row_sums = torch.zeros(N, device=config.device, dtype=config.dtype)
    row_counts = torch.zeros(N, device=config.device, dtype=config.dtype)

    row_sums.index_add_(0, row_indices, values)
    row_counts.index_add_(0, row_indices, torch.ones_like(values))

    row_counts = row_counts.clamp(min=1)
    row_means = row_sums / row_counts

    values -= row_means[row_indices]

    indices = torch.stack([row_indices, col_indices])
    weights = torch.sparse_coo_tensor(
        indices,
        values,
        size=(N, N),
        device=config.device,
        dtype=config.dtype,
    )

    return weights.coalesce().to_sparse_csr()


def create_er_sparse_cpu(config: ERGraphConfig) -> csr_matrix:
    """
    Create a sparse weighted Erdos-Renyi connectivity matrix

    This function follows the same logic as the function above
    but uses numpy and scipy instead of PyTorch because as of 03/2026
    PyTorch support for sparse operations on the CPU is limited
    and basic trials showed scipy producing results much faster.

    Returns:
        csr_matrix: compressed sparse row matrix of shape (N, N)
    """

    if config.device != torch.device("cpu"):
        raise ValueError(
            f"This function only supports cpu device, got {config.device=}"
        )

    if not isinstance(config.dtype, np.dtype):
        raise ValueError(
            f"This function only supports numpy datatypes, got {config.dtype=}"
        )

    N = config.num_neurons
    p = config.connection_prob

    expected_edges = int(N * N * p)

    row_indices = np.random.randint(0, N, size=expected_edges)
    col_indices = np.random.randint(0, N, size=expected_edges)

    values = np.random.randn(expected_edges)
    values *= _weight_scalar(config)

    row_sums = np.zeros(N, dtype=np.float32)
    row_counts = np.zeros(N, dtype=np.float32)

    np.add.at(row_sums, row_indices, values)
    np.add.at(row_counts, row_indices, 1)

    row_counts = np.maximum(row_counts, 1)
    row_means = row_sums / row_counts

    values -= row_means[row_indices]

    weights_csr = csr_matrix(
        (values, (row_indices, col_indices)), shape=(N, N), dtype=config.dtype
    )
    return weights_csr


def _weight_scalar(config: ERGraphConfig) -> float:
    """
    Return float for scaling weights during graph creation

    This function computes the following formula:

        g / (sqrt(N) * p)

    Rationale:
    - Expected in-degree ≈ pN
    - Variance of summed input should remain O(1) as N grows
    - sqrt(N) scaling keeps dynamics size-stable
    - Division by p compensates for sparsity

    This prevents trivial blow-up or vanishing activity as N varies,
    which is critical when measuring scaling laws.
    """

    return config.global_coupling_strength / (
        math.sqrt(config.num_neurons) * config.connection_prob
    )
