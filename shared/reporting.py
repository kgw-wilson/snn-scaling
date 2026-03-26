import torch
from shared.simulation_config import ERGraphConfig, SNNConfig


def report_spike_statistics(
    spikes_per_neuron: torch.Tensor,
    spikes_per_bin: torch.Tensor,
) -> None:
    """
    Print spike statistics after a simulation run

    Args:
        total_spike_count: Total number of spikes in the simulationn
        spikes_per_neuron: 1D tensor of shape [num_neurons] with spike counts per neuron
        spikes_per_bin: 1D tensor of shape [num_bins] with spike counts per time bin
    """

    print("== Spike Statistics ==")
    print(f"Total spikes: {spikes_per_neuron.sum().item()}")
    print(f"Mean spikes per neuron: {spikes_per_neuron.mean().item():.2f}")
    print(f"Std spikes per neuron: {spikes_per_neuron.std().item():.2f}")
    print(f"Mean spikes per bin: {spikes_per_bin.mean().item():.2f}")
    print(f"Std spikes per bin: {spikes_per_bin.std().item():.2f}")
    print(f"Max spikes by a neuron: {spikes_per_neuron.max().item()}")
    print(f"Min spikes by a neuron: {spikes_per_neuron.min().item()}")
    print(f"Max spikes in a bin: {spikes_per_bin.max().item()}")
    print(f"Min spikes in a bin: {spikes_per_bin.min().item()}")


def create_spike_reporting_tensors(
    graph_config: ERGraphConfig, snn_config: SNNConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize tensors used for spike-count statistics during simulation

    Assumes consumers of the function will bin indices when updating
    spikes_per_bin. No normalization is applied, values represent raw
    spike counts.

    Returns:
        spikes_per_neuron - [num_neurons] tracking total spike count per neuron
            over the full simulation duration

        spikes_per_bin - [num_bins] tracking aggregated spike activity over
            coarse time bins (useful for population firing rate analysis)
    """

    num_neurons = graph_config.num_neurons
    device = graph_config.device
    dtype = graph_config.dtype
    num_bins = snn_config.num_bins

    spikes_per_neuron = torch.zeros(num_neurons, device=device, dtype=dtype)
    spikes_per_bin = torch.zeros(num_bins, device=device, dtype=dtype)

    return spikes_per_neuron, spikes_per_bin
