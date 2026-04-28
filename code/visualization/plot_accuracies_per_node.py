import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils.logging import log_results


def extract_list(content: str, name: str):
    pattern = r"\*\*{}:\s*(\([^)]+\)(?:\s+\([^)]+\))*)".format(re.escape(name))
    match = re.search(pattern, content)
    if not match:
        return []
    data_str = "[" + match.group(1).replace(") (", "), (") + "]"
    return ast.literal_eval(data_str)


def build_per_node_series(value_tuples, cid_tuples):
    # Reconstruct per-node history even when client order changes each round.
    series = {}
    for (round_id, values), (cid_round, cids) in zip(value_tuples, cid_tuples):
        if round_id != cid_round:
            continue
        for value, cid in zip(values, cids):
            series.setdefault(cid, []).append((round_id, value))

    return series


def plot_metric(ax, series, ylabel: str, title: str, marker: str):
    for cid in sorted(series.keys()):
        points = sorted(series[cid], key=lambda item: item[0])
        rounds, values = zip(*points)
        ax.plot(rounds, values, marker=marker, label=f"Client {cid}")
    # Show only integer rounds on the x-axis.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python visualization/plot_accuracies_per_node.py <raw.out> [output.png]"
        )

    raw_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else raw_path.with_name("metrics_per_node.png")

    content = raw_path.read_text(encoding="utf-8")
    acc_distr = extract_list(content, "acc_distr")
    cid = extract_list(content, "cid")
    asr = extract_list(content, "asr")

    if not acc_distr or not cid:
        raise ValueError("raw.out does not contain acc_distr/cid tuples")

    node_acc = build_per_node_series(acc_distr, cid)
    node_asr = build_per_node_series(asr, cid) if asr else {}

    rows = 2 if node_asr else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 5 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    plot_metric(
        axes[0],
        node_acc,
        ylabel="Accuracy",
        title="Clean Accuracy per node",
        marker="o",
    )

    if node_asr:
        plot_metric(
            axes[1],
            node_asr,
            ylabel="ASR",
            title="Attack Success Rate per node",
            marker="s",
        )
        axes[1].set_xlabel("Round")
    else:
        axes[0].set_xlabel("Round")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    log_results(f"Saved per-node metrics plot to {output_path}", level="minimal")


if __name__ == "__main__":
    main()
