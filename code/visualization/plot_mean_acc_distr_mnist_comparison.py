import argparse
import ast
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt


ROUND_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\[[^\]]*\])\s*\)", re.DOTALL)
NUM_CLIENTS = 8


def configure_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


def extract_block(content: str, name: str) -> str | None:
    match = re.search(rf"\*\*{re.escape(name)}:\s*(.*?)(?=\n\*\*|\Z)", content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def parse_round_block(content: str, name: str) -> List[Tuple[int, list]]:
    block = extract_block(content, name)
    if block is None:
        return []

    parsed: List[Tuple[int, list]] = []
    matches = ROUND_PATTERN.findall(block)
    if not matches:
        return []

    for round_text, values_text in matches:
        try:
            round_id = int(round_text)
            values = ast.literal_eval(values_text)
        except (ValueError, SyntaxError):
            continue

        if not isinstance(values, list):
            continue

        parsed.append((round_id, values))

    parsed.sort(key=lambda item: item[0])
    return parsed


def parse_mean_acc_distr(content: str) -> List[Tuple[int, float]]:
    cid_rounds = parse_round_block(content, "cid")
    acc_rounds = parse_round_block(content, "acc_distr")

    if not cid_rounds or not acc_rounds:
        return []

    cid_map = {round_id: values for round_id, values in cid_rounds}
    acc_map = {round_id: values for round_id, values in acc_rounds}
    shared_rounds = sorted(set(cid_map).intersection(acc_map))

    global_state = {client_id: 0.0 for client_id in range(NUM_CLIENTS)}
    seen_clients = set()
    series: List[Tuple[int, float]] = []

    for round_id in shared_rounds:
        client_ids = cid_map[round_id]
        round_values = acc_map[round_id]

        for client_id, value in zip(client_ids, round_values):
            if client_id in global_state:
                global_state[client_id] = float(value)
                seen_clients.add(client_id)

        if seen_clients:
            mean_acc = sum(global_state[client_id] for client_id in seen_clients) / len(seen_clients)
        else:
            mean_acc = 0.0
        series.append((round_id, mean_acc))

    return series


def load_mean_acc_distr(file_path: Path) -> List[Tuple[int, float]]:
    if not file_path.exists():
        print(f"[WARN] Missing file: {file_path}")
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] Could not read file {file_path}: {exc}")
        return []

    mean_acc_distr = parse_mean_acc_distr(content)
    if not mean_acc_distr:
        print(f"[WARN] **acc_distr:/**cid: pattern not found or no valid rounds in {file_path}")
    return mean_acc_distr


def plot_series(ax, series: Sequence[Tuple[int, float]], label: str, color: str, marker: str, linestyle: str) -> None:
    if not series:
        return

    rounds, values = zip(*series)
    markevery = max(1, len(rounds) // 12)
    ax.plot(
        rounds,
        values,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.0,
        markersize=5.5,
        markevery=markevery,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a publication-ready Mean distributed accuracy comparison plot for MNIST topologies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--isolated", required=True, type=Path, help="Raw log file for the Isolated topology")
    parser.add_argument("--ring", required=True, type=Path, help="Raw log file for the Ring topology")
    parser.add_argument("--star", required=True, type=Path, help="Raw log file for the Star topology")
    parser.add_argument("--fc", required=True, type=Path, help="Raw log file for the Fully Connected topology")
    parser.add_argument("--sbm", required=True, type=Path, help="Raw log file for the SBM topology")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mean_acc_distr_mnist_comparison.png"),
        help="Output image path for the comparison plot",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_style()

    isolated_series = load_mean_acc_distr(args.isolated)
    ring_series = load_mean_acc_distr(args.ring)
    star_series = load_mean_acc_distr(args.star)
    fc_series = load_mean_acc_distr(args.fc)
    sbm_series = load_mean_acc_distr(args.sbm)

    if not any((isolated_series, ring_series, star_series, fc_series, sbm_series)):
        raise SystemExit("No valid **acc_distr: data found in any input file.")

    all_values = [value for series in (isolated_series, ring_series, star_series, fc_series, sbm_series) for _, value in series]
    min_value = min(all_values)
    max_value = max(all_values)
    y_padding = max(0.02, (max_value - min_value) * 0.08)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    plot_series(
        ax,
        isolated_series,
        label="Isolated",
        color="#1f77b4",
        marker="o",
        linestyle="-",
    )
    plot_series(
        ax,
        ring_series,
        label="Ring Clean",
        color="#ff7f0e",
        marker="s",
        linestyle="--",
    )
    plot_series(
        ax,
        star_series,
        label="Star Clean",
        color="#2ca02c",
        marker="^",
        linestyle=":",
    )
    plot_series(
        ax,
        fc_series,
        label="FC Clean",
        color="#d62728",
        marker="D",
        linestyle="-.",
    )
    plot_series(
        ax,
        sbm_series,
        label="SBM Clean",
        color="#9467bd",
        marker="v",
        linestyle=(0, (3, 1, 1, 1)),
    )

    ax.set_title("Evolution of Mean Distributed Accuracy across Topologies", pad=12)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Mean Distributed Accuracy")
    ax.set_yscale("linear")
    ax.set_xlim(0, 240)
    ax.set_ylim(max(0.0, min_value - y_padding), min(1.0, max_value + y_padding))
    ax.set_xticks(range(0, 241, 20))
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved Mean distributed accuracy comparison plot to {args.output}")


if __name__ == "__main__":
    main()