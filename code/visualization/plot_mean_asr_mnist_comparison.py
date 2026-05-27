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


def parse_mean_asr(content: str) -> List[Tuple[int, float]]:
    cid_rounds = parse_round_block(content, "cid")
    asr_rounds = parse_round_block(content, "asr")

    if not cid_rounds or not asr_rounds:
        return []

    cid_map = {round_id: values for round_id, values in cid_rounds}
    asr_map = {round_id: values for round_id, values in asr_rounds}
    shared_rounds = sorted(set(cid_map).intersection(asr_map))

    global_state = {client_id: 0.0 for client_id in range(NUM_CLIENTS)}
    series: List[Tuple[int, float]] = []

    for round_id in shared_rounds:
        client_ids = cid_map[round_id]
        round_asr_values = asr_map[round_id]

        for client_id, asr_value in zip(client_ids, round_asr_values):
            if client_id in global_state:
                global_state[client_id] = float(asr_value)

        mean_asr = sum(global_state.values()) / NUM_CLIENTS
        series.append((round_id, mean_asr))

    return series


def load_mean_asr(file_path: Path) -> List[Tuple[int, float]]:
    if not file_path.exists():
        print(f"[WARN] Missing file: {file_path}")
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] Could not read file {file_path}: {exc}")
        return []

    mean_asr = parse_mean_asr(content)
    if not mean_asr:
        print(f"[WARN] **asr:/**cid: pattern not found or no valid rounds in {file_path}")
    return mean_asr


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
        description="Generate a publication-ready Mean Attack Success Rate (ASR) comparison plot for MNIST topologies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fc", required=True, type=Path, help="Raw log file for the Fully Connected topology")
    parser.add_argument("--ring", required=True, type=Path, help="Raw log file for the Ring topology")
    parser.add_argument(
        "--star-hub",
        "--star_hub",
        dest="star_hub",
        required=True,
        type=Path,
        help="Raw log file for the Star hub topology",
    )
    parser.add_argument(
        "--star-periph",
        "--star_periph",
        dest="star_periph",
        required=True,
        type=Path,
        help="Raw log file for the Star peripheral topology",
    )
    parser.add_argument(
        "--sbm-gateway",
        "--sbm_gateway",
        dest="sbm_gateway",
        required=True,
        type=Path,
        help="Raw log file for the SBM gateway topology",
    )
    parser.add_argument(
        "--sbm-periph",
        "--sbm_periph",
        dest="sbm_periph",
        required=True,
        type=Path,
        help="Raw log file for the SBM peripheral topology",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mean_asr_mnist_comparison_all.png"),
        help="Output image path for the comparison plot",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_style()

    fc_series = load_mean_asr(args.fc)
    ring_series = load_mean_asr(args.ring)
    star_hub_series = load_mean_asr(args.star_hub)
    star_periph_series = load_mean_asr(args.star_periph)
    sbm_gateway_series = load_mean_asr(args.sbm_gateway)
    sbm_periph_series = load_mean_asr(args.sbm_periph)

    if not any((fc_series, ring_series, star_hub_series, star_periph_series, sbm_gateway_series, sbm_periph_series)):
        raise SystemExit("No valid **asr: data found in any input file.")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    plot_series(
        ax,
        fc_series,
        label="FC Topology",
        color="#1f77b4",
        marker="o",
        linestyle="-",
    )
    plot_series(
        ax,
        ring_series,
        label="Ring Topology",
        color="#ff7f0e",
        marker="s",
        linestyle="--",
    )
    plot_series(
        ax,
        star_hub_series,
        label="Star Hub",
        color="#2ca02c",
        marker="^",
        linestyle=":",
    )
    plot_series(
        ax,
        star_periph_series,
        label="Star Peripheral",
        color="#d62728",
        marker="D",
        linestyle="-.",
    )
    plot_series(
        ax,
        sbm_gateway_series,
        label="SBM Gateway",
        color="#9467bd",
        marker="v",
        linestyle="--",
    )
    plot_series(
        ax,
        sbm_periph_series,
        label="SBM Peripheral",
        color="#8c564b",
        marker="P",
        linestyle=":",
    )

    ax.set_title("Evolution of Mean Attack Success Rate across Topologies", pad=12)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Mean Attack Success Rate (ASR)")
    ax.set_xlim(0, 240)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(0, 241, 20))
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved Mean ASR comparison plot to {args.output}")


if __name__ == "__main__":
    main()