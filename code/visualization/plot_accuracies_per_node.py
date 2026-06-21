import sys
from pathlib import Path

# Force the working directory where you ran the command into Python's path search
if "" not in sys.path:
    sys.path.insert(0, "")

import ast
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Now import your utility module safely
try:
    from utils.logging import log_results
except ModuleNotFoundError:
    # Fallback if the path environment is locked down
    def log_results(msg, level="minimal"):
        print(f"[{level.upper()}] {msg}")

# --- THESIS PLOT CONFIGURATION (Larger Text) ---
plt.rcParams.update({
    'font.size': 20,          # Base size
    'axes.titlesize': 20,     # Plot titles
    'axes.labelsize': 20,     # X and Y axis labels
    'xtick.labelsize': 18,    # X axis tick labels
    'ytick.labelsize': 18,    # Y axis tick labels
    'legend.fontsize': 18     # Legend text
})

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]


def extract_list(content: str, name: str):
    pattern = r"\*\*{}:\s*(\([^)]+\)(?:\s+\([^)]+\))*)".format(re.escape(name))
    match = re.search(pattern, content)
    if not match:
        return []
    data_str = "[" + match.group(1).replace(") (", "), (") + "]"
    return ast.literal_eval(data_str)


def build_per_node_series(value_tuples, cid_tuples):
    series = {}
    for (round_id, values), (cid_round, cids) in zip(value_tuples, cid_tuples):
        if round_id != cid_round:
            continue
        for value, cid in zip(values, cids):
            series.setdefault(cid, []).append((round_id, value))
    return series


def plot_metric(ax, series, ylabel: str, title: str, marker: str, ymin: float = None):
    for index, cid in enumerate(sorted(series.keys())):
        points = sorted(series[cid], key=lambda item: item[0])
        rounds, values = zip(*points)
        point_marker = MARKERS[index % len(MARKERS)] if marker == "auto" else marker
        ax.plot(
            rounds,
            values,
            marker=point_marker,
            linestyle="-",
            linewidth=1.5,      # Slightly thicker lines for visibility in thesis
            markeredgewidth=1.0,
            label=f"Client {cid}",
        )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymin is not None:
        ax.set_ylim(ymin, 1.00)  # Slightly elevated max to fit top points cleanly
    else:
        ax.set_ylim(-0.05, 1.00)
    ax.grid(True, linestyle='--', alpha=0.7)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python visualization/plot_accuracies_per_node.py <raw.out> [output.png]"
        )

    raw_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else raw_path.with_name("metrics_per_node.png")

    print(f"--- Processing: {raw_path.name} ---")

    content = raw_path.read_text(encoding="utf-8")
    acc_distr = extract_list(content, "acc_distr")
    cid = extract_list(content, "cid")
    asr = extract_list(content, "asr")

    if not acc_distr or not cid:
        raise ValueError("raw.out does not contain acc_distr/cid tuples")

    node_acc = build_per_node_series(acc_distr, cid)
    node_asr = build_per_node_series(asr, cid) if asr else {}

    # --- SIDE-BY-SIDE GRID LAYOUT ---
    cols = 2 if node_asr else 1
    fig, axes = plt.subplots(1, cols, figsize=(11 * cols, 5.5))
    if cols == 1:
        axes = [axes]

    # Plot 1: Clean Accuracy
    plot_metric(
        axes[0],
        node_acc,
        ylabel="Accuracy",
        title="Clean Accuracy per Node",
        marker="auto",
        ymin=0.80
    )
    axes[0].set_xlabel("Round")

    # Plot 2: Attack Success Rate (if present)
    if node_asr:
        plot_metric(
            axes[1],
            node_asr,
            ylabel="ASR",
            title="Attack Success Rate per Node",
            marker="auto",
        )
        axes[1].set_xlabel("Round")
        
        # Place single combined legend outside the right-most plot
        axes[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
    else:
        axes[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)

    # Use constrained layout engine to prevent outside legend clipping
    fig.set_layout_engine('constrained')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with high resolution (300 DPI) for printing/thesis text crispness
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"SUCCESS: Saved plot to {output_path}")
    try:
        log_results(f"Saved per-node metrics plot to {output_path}", level="minimal")
    except Exception:
        pass


if __name__ == "__main__":
    main()