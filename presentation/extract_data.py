"""Read-only extraction of logged results into deck_data.json (runs no simulations).

Sources:
  logs/04-30/results.csv  -- main sweep. Rows 2-37 (file lines) are the first
      'clock_driven_dense,cpu' block; rows 38-67 are a second block with the same
      label that scales like event-driven and is EXCLUDED here (see report).
  git show eb02eb9:results.csv -- dense CPU vs Brian2 (read from the commit so any branch works).
  git show 084dd31:results.csv -- deleted OpenMP backend + event-driven on same run.
  logs/04-16/output.txt -- eager vs torch.compile (Colab).
"""
import csv
import json
import statistics as st
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = {}

rows = list(csv.DictReader(open(ROOT / "logs/04-30/results.csv")))
# file line = index + 2. First dense-cpu block = file lines 2..37 -> idx 0..35
label = {
    ("clock_driven_dense", "cpu"): "Dense CPU",
    ("clock_driven_dense", "gpu"): "Dense GPU",
    ("clock_driven_sparse_cpu", "cpu"): "Sparse CPU",
    ("clock_driven_sparse_gpu", "gpu"): "Sparse GPU",
    ("event_driven", "cpu"): "Event-driven CPU",
}
sweep = {}
for i, r in enumerate(rows):
    key = (r["func_name"], r["device_str"])
    if key == ("clock_driven_dense", "cpu") and not (0 <= i <= 35):
        continue  # second, mislabeled block
    name = label[key]
    sweep.setdefault(name, []).append({
        "N": int(r["num_neurons"]), "p": float(r["connection_prob"]),
        "t": float(r["elapsed_time"]), "spn": float(r["mean_spikes_per_neuron"]),
    })
out["sweep"] = sweep
out["excluded_block"] = [
    {"N": int(r["num_neurons"]), "p": float(r["connection_prob"]), "t": float(r["elapsed_time"])}
    for i, r in enumerate(rows)
    if (r["func_name"], r["device_str"]) == ("clock_driven_dense", "cpu") and i > 35
]

# Brian2 vs dense (HEAD)
b_txt = subprocess.run(["git", "show", "eb02eb9:results.csv"], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout
b = list(csv.DictReader(b_txt.splitlines()))
br = {}
for r in b:
    br.setdefault(r["func_name"], {}).setdefault(r["connection_prob"], []).append(
        (float(r["elapsed_time"]), float(r["mean_spikes_per_neuron"])))
out["brian2"] = {
    f: {p: {"t_mean": st.mean(x[0] for x in v), "t_all": [x[0] for x in v],
            "spn_mean": st.mean(x[1] for x in v), "spn_all": [x[1] for x in v]}
        for p, v in d.items()} for f, d in br.items()}

# OpenMP run (deleted; from git history)
txt = subprocess.run(["git", "show", "084dd31:results.csv"], cwd=ROOT,
                     capture_output=True, text=True, check=True).stdout
omp = {}
for r in csv.DictReader(txt.splitlines()):
    omp.setdefault(r["func_name"], []).append(
        {"N": int(r["num_neurons"]), "p": float(r["connection_prob"]),
         "t": float(r["elapsed_time"]), "spn": float(r["mean_spikes_per_neuron"])})
out["openmp_run"] = omp

# torch.compile (04-16)
lines = (ROOT / "logs/04-16/output.txt").read_text().split("===================")
tc = {}
for sec in lines:
    s = sec.strip().splitlines()
    if not s or not s[0].startswith("clock_driven"):
        continue
    head = s[0].split(", ")
    took = [line for line in s if line.startswith("Simulation main took")]
    if not took:
        continue
    tc.setdefault(head[0] + "|" + head[1], []).append(float(took[0].split()[3]))
out["torch_compile"] = tc

(Path(__file__).parent / "deck_data.json").write_text(json.dumps(out, indent=1))
print("wrote deck_data.json")


# ---------------- derived numbers (arithmetic on logged values only) ----------------
def get(name, N, p):
    for r in sweep[name]:
        if r["N"] == N and abs(r["p"] - p) < 1e-9:
            return r
    return None

Ns = [10, 100, 1000, 10000]
ps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
derived = {}

# winner map
winner = {}
for N in Ns:
    for p in ps:
        cands = [(get(n, N, p)["t"], n) for n in sweep if get(n, N, p)]
        t, n = min(cands)
        second = sorted(cands)[1]
        winner[f"{N}|{p}"] = {"winner": n, "t": t, "runner_up": second[1], "ratio": second[0] / t}
derived["winner"] = winner

# cross-backend agreement in mean spikes/neuron (only cells where all 5 exist, N >= 1000)
clock = ["Dense CPU", "Dense GPU", "Sparse CPU", "Sparse GPU"]
spreads, ev_gaps = [], []
for N in (1000, 10000):
    for p in ps:
        vals = [get(n, N, p)["spn"] for n in clock]
        m = sum(vals) / 4
        spreads.append((max(vals) - min(vals)) / m * 100)
        e = get("Event-driven CPU", N, p)
        if e:
            ev_gaps.append({"N": N, "p": p, "gap_pct": (e["spn"] - m) / m * 100})
derived["clock_spread_max_pct"] = max(spreads)
derived["event_gap_pct"] = ev_gaps

# event-driven cost per synaptic event: runtime / (N * spikes_per_neuron * N*p + N*50Hz*1s Poisson)
pe = []
for r in sweep["Event-driven CPU"]:
    if r["N"] >= 1000:
        events = r["N"] * r["spn"] * int(r["N"] * r["N"] * r["p"]) / r["N"] + r["N"] * 50
        pe.append({"N": r["N"], "p": r["p"], "events": events, "ns_per_event": r["t"] / events * 1e9})
derived["event_ns_per_event"] = pe

# dense GPU effective bandwidth at N=1e4: weights = 3 buckets * N^2 * 4 B read per step, 1000 steps
t = get("Dense GPU", 10000, 1.0)["t"]
derived["denseGPU_N1e4_GBps"] = 3 * 1e8 * 4 * 1000 / t / 1e9
derived["denseGPU_floor_ms_per_step"] = [get("Dense GPU", 1000, p)["t"] for p in ps]

# brian2 relative differences
b2 = out["brian2"]
derived["brian2_diff_pct"] = {p: (b2["clock_driven_dense"][p]["spn_mean"] - b2["clock_driven_brian2"][p]["spn_mean"])
                               / b2["clock_driven_brian2"][p]["spn_mean"] * 100 for p in b2["clock_driven_dense"]}
out["derived"] = derived
(Path(__file__).parent / "deck_data.json").write_text(json.dumps(out, indent=1))
print("wrote derived numbers")
