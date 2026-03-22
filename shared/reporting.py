from typing import Union
import numpy as np
import torch


def report_spike_statistics(
    spikes_per_neuron: Union[torch.Tensor, np.ndarray],
    spikes_per_bin: Union[torch.Tensor, np.ndarray],
) -> None:
    """
    Print spike statistics after a simulation run

    Args:
        total_spike_count: Total number of spikes in the simulationn
        spikes_per_neuron: 1D tensor of shape [num_neurons] with spike counts per neuron
        spikes_per_bin: 1D tensor of shape [num_bins] with spike counts per time bin
    """

    if isinstance(spikes_per_neuron, torch.Tensor):
        spikes_per_neuron = spikes_per_neuron.cpu().numpy()
    if isinstance(spikes_per_bin, torch.Tensor):
        spikes_per_bin = spikes_per_bin.cpu().numpy()

    mean_per_neuron = spikes_per_neuron.mean()
    std_per_neuron = spikes_per_neuron.std()
    mean_per_bin = spikes_per_bin.mean()
    std_per_bin = spikes_per_bin.std()

    max_spikes_neuron = spikes_per_neuron.max()
    min_spikes_neuron = spikes_per_neuron.min()
    max_spikes_bin = spikes_per_bin.max()
    min_spikes_bin = spikes_per_bin.min()

    print("== Spike Statistics ==")
    print(f"Total spikes: {spikes_per_neuron.sum()}")
    print(f"Mean spikes per neuron: {mean_per_neuron:.2f}")
    print(f"Std spikes per neuron: {std_per_neuron:.2f}")
    print(f"Mean spikes per bin: {mean_per_bin:.2f}")
    print(f"Std spikes per bin: {std_per_bin:.2f}")
    print(f"Max spikes by a neuron: {max_spikes_neuron}")
    print(f"Min spikes by a neuron: {min_spikes_neuron}")
    print(f"Max spikes in a bin: {max_spikes_bin}")
    print(f"Min spikes in a bin: {min_spikes_bin}")
