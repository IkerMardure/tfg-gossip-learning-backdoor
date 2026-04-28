import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from utils.logging import log_results


def extract_list(content: str, name: str):
    pattern = r"\*\*{}:\s*(\([^)]+\)(?:\s+\([^)]+\))*)".format(re.escape(name))
    match = re.search(pattern, content)
    if not match:
        return []
    data_str = "[" + match.group(1).replace(") (", "), (") + "]"
    return ast.literal_eval(data_str)


def load_metrics(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    losses_distributed = extract_list(content, "losses_distributed")
    losses_centralized = extract_list(content, "losses_centralized")
    acc_distr = extract_list(content, "acc_distr")
    metrics_centralized = extract_list(content, "metrics_centralized")
    asr = extract_list(content, "asr")

    avg_acc_distr = []
    for round_id, values in acc_distr:
        if values:
            avg_acc_distr.append((round_id, sum(values) / len(values)))

    avg_asr = []
    for round_id, values in asr:
        if values:
            avg_asr.append((round_id, sum(values) / len(values)))

    return {
        "losses_distributed": losses_distributed,
        "losses_centralized": losses_centralized,
        "avg_acc_distr": avg_acc_distr,
        "metrics_centralized": metrics_centralized,
        "avg_asr": avg_asr,
    }


def plot_series(ax, series, label, marker):
    if not series:
        return
    rounds, values = zip(*series)
    ax.plot(rounds, values, marker=marker, label=label)


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python visualization/plot_clean_vs_backdoor.py <clean_raw.out> <backdoor_raw.out> <output.png>"
        )

    clean_path = Path(sys.argv[1])
    backdoor_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    clean = load_metrics(clean_path)
    backdoor = load_metrics(backdoor_path)

    has_asr = bool(backdoor["avg_asr"])
    rows = 4 if has_asr else 3
    fig, axes = plt.subplots(rows, 1, figsize=(11, 4 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    plot_series(axes[0], clean["avg_acc_distr"], "Clean avg distributed acc", "o")
    plot_series(axes[0], backdoor["avg_acc_distr"], "Backdoor avg distributed acc", "x")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Distributed accuracy")
    axes[0].grid(True)
    axes[0].legend()

    plot_series(axes[1], clean["metrics_centralized"], "Clean centralized acc", "o")
    plot_series(axes[1], backdoor["metrics_centralized"], "Backdoor centralized acc", "x")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Centralized accuracy")
    axes[1].grid(True)
    axes[1].legend()

    plot_series(axes[2], clean["losses_distributed"], "Clean distributed loss", "o")
    plot_series(axes[2], backdoor["losses_distributed"], "Backdoor distributed loss", "x")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Distributed loss")
    axes[2].grid(True)
    axes[2].legend()

    if has_asr:
        plot_series(axes[3], backdoor["avg_asr"], "Backdoor avg ASR", "s")
        axes[3].set_ylabel("ASR")
        axes[3].set_title("Attack Success Rate")
        axes[3].grid(True)
        axes[3].legend()
        axes[3].set_xlabel("Round")
    else:
        axes[2].set_xlabel("Round")

    fig.suptitle("Clean vs Backdoor comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    log_results(f"Comparison plot saved to {output_path}", level="minimal")


if __name__ == "__main__":
    main()