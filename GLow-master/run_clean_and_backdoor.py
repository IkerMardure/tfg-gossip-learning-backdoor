import subprocess
import sys
from pathlib import Path

import yaml


def _run_step(label: str, script_name: str, conf_file: str, run_id: str, topology_file: str) -> None:
    cmd = [sys.executable, script_name, conf_file, run_id, topology_file]
    print(f"\n[{label}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[{label}] failed with exit code {result.returncode}")
    print(f"[{label}] completed successfully")


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python run_clean_and_backdoor.py <conf_file> <run_prefix> <topology_file>"
        )

    conf_file = sys.argv[1]
    run_prefix = sys.argv[2]
    topology_file = sys.argv[3]

    script_dir = Path(__file__).resolve().parent

    with open(conf_file, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    clean_script = str(script_dir / "main.py")
    backdoor_script = str(script_dir / "main_backdoor.py")
    plot_script = str(script_dir / "visualization" / "plot_clean_vs_backdoor.py")

    clean_run_id = f"{run_prefix}_clean"
    backdoor_run_id = f"{run_prefix}_backdoor"
    output_dir = script_dir / "outputs" / cfg["run_name"]
    clean_raw = output_dir / f"{clean_run_id}_raw.out"
    backdoor_raw = output_dir / f"{backdoor_run_id}_raw.out"
    comparison_plot = output_dir / f"{run_prefix}_clean_vs_backdoor.png"

    print("Running paired experiment (clean -> backdoor) with same config and topology")
    _run_step("BASELINE", clean_script, conf_file, clean_run_id, topology_file)
    _run_step("BACKDOOR", backdoor_script, conf_file, backdoor_run_id, topology_file)

    _run_step(
        "PLOT",
        plot_script,
        str(clean_raw),
        str(backdoor_raw),
        str(comparison_plot),
    )

    print("\nPaired run finished")
    print(f"- clean run id: {clean_run_id}")
    print(f"- backdoor run id: {backdoor_run_id}")
    print(f"- comparison plot: {comparison_plot}")


if __name__ == "__main__":
    main()