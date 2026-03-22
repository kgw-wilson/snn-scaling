import math
from typing import Union
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


def create_er_sparse(
    config: ERGraphConfig, use_numpy: bool
) -> Union[torch.Tensor, csr_matrix]:
    """
    Create a sparse weighted Erdos-Renyi connectivity matrix

    This generates a sparse weight matrix in PyTorch using the same
    random sampling and scaling logic as the create_er_dense function.

    Callers of this function should set use_numpy to False if running
    a simulation with a CUDA backend. Callers should set use_numpy to
    True if running on CPU or MPS. This is because PyTorch's sparse support
    on CPU and MPS is currently inefficient or nonexistent, respectively
    (as of 03/2026).

    The weight tensor is created as coo first because of its
    more human-friendly format and then converted to csr for
    faster computation.

    Returns:
        torch.Tensor - sparse csr tensor of shape (N, N) if use_numpy is False
        csr_matrix - sparse csr matrix of shape (N, N) if use_numpy is True
    """

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
    weights_coo = torch.sparse_coo_tensor(
        indices,
        values,
        size=(N, N),
        device=config.device,
        dtype=config.dtype,
    )
    weights_csr = weights_coo.coalesce().to_sparse_csr()

    if use_numpy:
        row_ind = weights_csr.crow_indices().numpy()
        col_ind = weights_csr.col_indices().numpy()
        data = weights_csr.values().numpy()
        return csr_matrix((data, col_ind, row_ind), shape=(N, N))

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
