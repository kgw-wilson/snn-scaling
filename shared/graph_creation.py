from typing import Union
from scipy.sparse import csr_matrix
import torch
from shared.simulation_config import SimulationConfig


def create_er_dense(sim_config: SimulationConfig) -> torch.Tensor:
    """
    Create a dense weighted Erdos-Renyi connectivity matrix

    This function generates a dense N x N weight matrix W suitable for
    benchmarking SNN simulation performance on CPU and GPU.

    Each directed edge is included independently with probability p and
    no structural bias (i.e. no small-world or scale-free structure).
    This isolates scaling behavior with respect to N and p. Implementation
    uses a random permutation and generates the exact number of expected edges.

    Returns:
        torch.Tensor - dense weight matrix of shape N x N on specified device
    """

    N = sim_config.num_neurons
    p = sim_config.connection_prob

    torch.manual_seed(int(N * p)) # TODO: put in for testing

    num_edges = int(N * N * p)

    weights = torch.full(
        (sim_config.num_neurons, sim_config.num_neurons),
        fill_value=sim_config.recurrent_weight,
        device=sim_config.device,
        dtype=sim_config.dtype,
    )

    all_indices = torch.arange(N * N, device=sim_config.device)
    chosen_indices = all_indices[torch.randperm(N * N)[:num_edges]]
    rows = chosen_indices // N
    cols = chosen_indices % N

    connectivity_mask = torch.zeros((N, N), dtype=torch.bool, device=sim_config.device)
    connectivity_mask[rows, cols] = True

    weights = weights * connectivity_mask

    return weights


def create_er_sparse(
    sim_config: SimulationConfig, use_numpy: bool
) -> Union[torch.Tensor, csr_matrix]:
    """
    Create a sparse weighted Erdos-Renyi connectivity matrix

    This generates a sparse weight matrix in PyTorch using similar
    logic as the create_er_dense function. This requires edges to
    be unique, as .coalesce() sums duplicate edges which breaks the
    expectation of constant weight values.

    Callers of this function should set use_numpy to False if running
    a simulation with a CUDA backend. Callers should set use_numpy to
    True if running on CPU. This is because PyTorch's sparse support
    on CPU is inefficient (as of 03/2026).

    The weight tensor is created as coo first because of its
    more human-friendly format and then converted to csr for
    faster computation.

    Returns:
        torch.Tensor - sparse csr tensor of shape (N, N) if use_numpy is False

        csr_matrix - sparse csr matrix of shape (N, N) if use_numpy is True
    """

    N = sim_config.num_neurons
    p = sim_config.connection_prob
    device = sim_config.device
    dtype = sim_config.dtype

    torch.manual_seed(int(N * p)) # TODO: put in for testing

    num_edges = int(N * N * p)

    all_indices = torch.arange(N * N, device=sim_config.device)
    chosen_indices = all_indices[torch.randperm(N * N)[:num_edges]]
    row_indices = chosen_indices // N
    col_indices = chosen_indices % N

    values = torch.full(
        (num_edges,),
        fill_value=sim_config.recurrent_weight,
        device=device,
        dtype=dtype,
    )

    weights_coo = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        values,
        size=(N, N),
        device=device,
        dtype=dtype,
    )
    weights_csr = weights_coo.coalesce().to_sparse_csr()

    if use_numpy:
        row_ind = weights_csr.crow_indices().numpy()
        col_ind = weights_csr.col_indices().numpy()
        data = weights_csr.values().numpy()
        return csr_matrix((data, col_ind, row_ind), shape=(N, N))

    return weights_csr
