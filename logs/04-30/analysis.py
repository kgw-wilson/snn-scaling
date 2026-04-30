import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results_path: str = "results.csv"):
    df = pd.read_csv(results_path)

    # Define the 3 cases explicitly
    cases = [
        ("clock_driven_dense", "cpu", "Clock-Driven Dense (CPU)"),
        ("clock_driven_dense", "gpu", "Clock-Driven Dense (GPU)"),
        ("clock_driven_sparse_gpu", "gpu", "Clock-Driven Sparse (GPU)"),
        ("event_driven", "cpu", "Event-Driven"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharex=True, sharey=True)
    fig.suptitle("Runtime vs Connection Probability", fontsize=14)

    cmap = plt.get_cmap("tab10")

    for ax, (func_name, device_str, title) in zip(axes, cases):

        # Filter per case
        if device_str is None:
            df_case = df[df["func_name"] == func_name]
        else:
            df_case = df[
                (df["func_name"] == func_name) &
                (df["device_str"] == device_str)
            ]

        if df_case.empty:
            ax.set_title(f"{title}\n(No data)")
            continue

        # Color by num_neurons for consistency within each subplot
        num_neurons_list = sorted(df_case["num_neurons"].unique())
        color_map = {
            n: cmap(i % 10) for i, n in enumerate(num_neurons_list)
        }

        # Group and plot
        for num_neurons, group in df_case.groupby("num_neurons"):

            group = (
                group.groupby("connection_prob", as_index=False)["elapsed_time"]
                .mean()
                .sort_values("connection_prob")
            )

            ax.plot(
                group["connection_prob"],
                group["elapsed_time"],
                marker="o",
                linestyle="-",
                color=color_map[num_neurons],
                label=f"N={num_neurons}",
            )

        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Connection Probability (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Runtime (log scale)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(results_path).parent / "runtime_vs_connection_prob.png"
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_results()