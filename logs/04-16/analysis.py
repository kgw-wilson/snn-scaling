import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


with open("output.txt", "r") as f:
    file_text = f.read()

separator = "==================="

sections = file_text.split(separator)

file_data = []

_NUM_REPEATS = 5

for i, section in enumerate(sections):
    if i % _NUM_REPEATS == 0 or i % _NUM_REPEATS == 1:
        continue
    lines = section.strip().split("\n")
    sim_name, n_text, p_text = lines[0].split(", ")
    num_neurons = int(n_text.split("=")[1])
    connection_prob = float(p_text.split("=")[1])
    runtime = float(lines[1].split(" ")[3])
    num_spikes = float(lines[2].split("=")[1])
    line_dict = {
        "sim_name": sim_name,
        "num_neurons": num_neurons,
        "connection_prob": connection_prob,
        "runtime": runtime,
        "num_spikes": num_spikes,
    }
    file_data.append(line_dict)

df = pd.DataFrame(file_data)

# Create x-axis
df["effective_connections"] = df["num_neurons"] * df["connection_prob"]

# Remove zeros if using log scale
df = df[(df["effective_connections"] > 0) & (df["runtime"] > 0)]

sim_names = df["sim_name"].unique()

plt.figure()

for sim in sim_names:
    subset = df[df["sim_name"] == sim]
    
    plt.scatter(
        subset["effective_connections"],
        subset["runtime"],
        alpha=0.2
    )
    
    grouped = (
        subset
        .groupby("effective_connections")["runtime"]
        .mean()
        .reset_index()
        .sort_values("effective_connections")
    )
    
    x = grouped["effective_connections"].values
    y = grouped["runtime"].values
    
    # --- Fit line in log-log space (power law fit) ---
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    coeffs = np.polyfit(log_x, log_y, 1)  # slope, intercept
    slope, intercept = coeffs
    
    # Generate smooth fit line
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_fit = 10**(intercept) * x_fit**slope
    
    # --- Plot fit line (same color as scatter) ---
    line, = plt.plot(x_fit, y_fit, label=f"{sim} (slope={slope:.2f})")
    
    # Match scatter color to line color
    plt.scatter(
        subset["effective_connections"],
        subset["runtime"],
        alpha=0.15,
        color=line.get_color()
    )

# Log scales (important for scaling analysis)
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Effective Connections (log scale)")
plt.ylabel("Runtime (log scale)")
plt.title("Runtime Scaling with Connectivity")
plt.legend()
# plt.grid(True, which="both")

plt.show()