import torch
from shared.simulation_config import SimulationConfig


import csv
from pathlib import Path


def report_statistics(
    sim_config: SimulationConfig,
    func_name: str,
    elapsed_time: float,
    spikes_per_neuron: torch.Tensor,
    spikes_per_bin: torch.Tensor,
) -> None:
    """
    Save statistics to CSV after a simulation run

    Args:
        sim_config: SimulationConfig contains simulation parameters
        func_name: name of the simulation function/backend
        elapsed_time: float representing total runtime of the simulation in seconds
        spikes_per_neuron: int tensor of shape [num_neurons] with spike counts per neuron
        spikes_per_bin: int tensor of shape [num_bins] with spike counts per time bin
    """
    results_path = Path("results.csv")

    row = {
        "func_name": func_name,
        "elapsed_time": elapsed_time,
        "device": sim_config.device_str,
        "num_neurons": sim_config.num_neurons,
        "connection_prob": sim_config.connection_prob,
        "timestep": sim_config.timestep,
        "min_delay": sim_config.min_delay,
        "max_delay": sim_config.max_delay,
        "mean_spikes_per_neuron": spikes_per_neuron.float().mean().item(),
        "mean_spikes_per_bin": spikes_per_bin.float().mean().item(),
    }

    write_header = not results_path.exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)



def create_spike_reporting_tensors(
    sim_config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize tensors used for spike-count statistics during simulation

    Assumes consumers of the function will bin indices when updating
    spikes_per_bin. No normalization is applied, values represent raw
    spike counts.

    Returns:
        spikes_per_neuron - int tensor [num_neurons] tracking total spike
            count per neuron over the full simulation duration

        spikes_per_bin - int tensor [num_bins] tracking aggregated spike
            activity over coarse time bins (useful for population firing
            rate analysis)
    """

    num_neurons = sim_config.num_neurons
    device = sim_config.device
    num_bins = sim_config.num_bins

    spikes_per_neuron = torch.zeros(num_neurons, device=device, dtype=torch.int32)
    spikes_per_bin = torch.zeros(num_bins, device=device, dtype=torch.int32)

    return spikes_per_neuron, spikes_per_bin
