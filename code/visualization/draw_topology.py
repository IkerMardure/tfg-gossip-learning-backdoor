import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.topology_visualization import visualize_topology_from_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a topology graph image from a GLow YAML topology file."
    )
    parser.add_argument("--yaml", required=True, help="Path to topology YAML file")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file or output directory. "
            "If omitted, output is saved to visualization/ as <name>_graph.<format>."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--layout",
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "shell"],
        help="Graph layout algorithm (default: spring)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic layouts")
    parser.add_argument("--dpi", type=int, default=250, help="Image DPI (default: 250)")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide node and community labels",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    
    # Default output to visualization/ if not specified
    output_path = args.out if args.out else "visualization"
    
    output_file = visualize_topology_from_yaml(
        yaml_path=args.yaml,
        output_path=output_path,
        image_format=args.format,
        layout=args.layout,
        seed=args.seed,
        dpi=args.dpi,
        show_labels=not args.no_labels,
    )
    print(f"✓ Graph image generated: {output_file}")


if __name__ == "__main__":
    main()
