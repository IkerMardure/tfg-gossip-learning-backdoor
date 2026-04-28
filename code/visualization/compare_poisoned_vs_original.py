import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import yaml

from utils.paths import resolve_data_path

from client_backdoor import BackdoorDataset
from dataset import prepare_dataset_iid, prepare_dataset_mnist_iid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare original and backdoored samples side-by-side and save as PNG."
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to load (default: mnist)",
    )
    parser.add_argument(
        "--topology-yaml",
        default=None,
        help="Optional topology YAML to read num_clients and clients_with_no_data",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client ID from which to sample data (default: 0)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Used when --topology-yaml is not provided (default: 5)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used by dataset preparation (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible split and sample selection",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=0,
        help="Backdoor target class (default: 0)",
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=0.5,
        help="Ratio of samples to poison in wrapped dataset (default: 0.5)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of clean/poisoned pairs to render (default: 8)",
    )
    parser.add_argument(
        "--sample-mode",
        default="poisoned",
        choices=["poisoned", "mixed"],
        help="Pick only poisoned indices or random mixed indices (default: poisoned)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Dataset root path for MNIST loader (default: project data/datasets)",
    )
    parser.add_argument(
        "--output",
        default="visualization/poisoned_vs_original.png",
        help="Output PNG path (default: visualization/poisoned_vs_original.png)",
    )
    parser.add_argument(
        "--view-mode",
        default="network",
        choices=["network", "denormalized"],
        help=(
            "How to render tensors: 'network' shows exact normalized input used by the model; "
            "'denormalized' shows human-readable pixels"
        ),
    )
    return parser


def _load_topology_info(topology_yaml: str, num_clients: int) -> Tuple[int, List[int]]:
    if not topology_yaml:
        return num_clients, []

    yaml_path = Path(topology_yaml)
    with yaml_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    yaml_num_clients = int(cfg.get("num_clients", num_clients))
    clients_with_no_data = [int(cid) for cid in cfg.get("clients_with_no_data", [])]
    return yaml_num_clients, clients_with_no_data


def _denormalize_for_plot(image: torch.Tensor, dataset_name: str) -> torch.Tensor:
    x = image.detach().cpu().clone()
    if dataset_name == "mnist":
        mean = torch.tensor([0.1307]).view(1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    x = x * std + mean
    return torch.clamp(x, 0.0, 1.0)


def _choose_indices(pool: Sequence[int], num_samples: int, seed: int) -> Sequence[int]:
    if len(pool) == 0:
        return []
    count = min(num_samples, len(pool))
    rng = random.Random(seed)
    return sorted(rng.sample(list(pool), count))


def _load_client_trainset(args: argparse.Namespace):
    num_clients, clients_with_no_data = _load_topology_info(args.topology_yaml, args.num_clients)
    if args.client_id < 0 or args.client_id >= num_clients:
        raise ValueError(f"--client-id must be in [0, {num_clients - 1}]")

    if args.dataset == "mnist":
        resolved = resolve_data_path(args.data_path)
        trainloaders, _, _ = prepare_dataset_mnist_iid(
            num_clients=num_clients,
            num_classes=args.num_classes,
            clients_with_no_data=clients_with_no_data,
            batch_size=args.batch_size,
            seed=args.seed,
            data_path=str(resolved),
        )
    else:
        trainloaders, _, _ = prepare_dataset_iid(
            num_clients=num_clients,
            num_classes=args.num_classes,
            clients_with_no_data=clients_with_no_data,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    client_loader = trainloaders[args.client_id]
    if isinstance(client_loader, str) or client_loader == "":
        raise ValueError(f"Client {args.client_id} has no data in this topology split")
    return client_loader.dataset


def _render_image(ax, image: torch.Tensor, dataset_name: str, view_mode: str) -> None:
    if view_mode == "denormalized":
        image = _denormalize_for_plot(image, dataset_name)

    if dataset_name == "mnist":
        if view_mode == "network":
            ax.imshow(image.squeeze(0), cmap="gray", vmin=-0.5, vmax=1.0)
        else:
            ax.imshow(image.squeeze(0), cmap="gray")
    else:
        if view_mode == "network":
            # Keep consistent contrast for normalized CIFAR tensors.
            show_img = torch.clamp(image, -1.0, 1.0)
            show_img = (show_img + 1.0) / 2.0
            ax.imshow(show_img.permute(1, 2, 0))
        else:
            ax.imshow(image.permute(1, 2, 0))


def _render_comparison(
    clean_dataset,
    poisoned_dataset,
    indices: Sequence[int],
    dataset_name: str,
    target_class: int,
    output_path: Path,
    view_mode: str,
) -> None:
    n_rows = len(indices)
    if n_rows == 0:
        raise ValueError("No samples available to visualize")

    fig, axes = plt.subplots(n_rows, 2, figsize=(8, max(3, n_rows * 2.2)))
    if n_rows == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        clean_img, clean_label = clean_dataset[idx]
        poison_img, poison_label = poisoned_dataset[idx]
        is_poisoned_index = idx in poisoned_dataset.poisoned_indices

        left_ax = axes[row][0]
        right_ax = axes[row][1]

        _render_image(left_ax, clean_img, dataset_name, view_mode)
        _render_image(right_ax, poison_img, dataset_name, view_mode)

        left_ax.set_title(f"Original | y={int(clean_label)}")
        right_title = f"Poisoned | y={int(poison_label)}"
        if is_poisoned_index:
            right_title += " | trigger:on"
            # Matches client_backdoor.py exactly: x_poisoned[0, 25:, 25:] = 1.0
            right_ax.add_patch(
                patches.Rectangle(
                    (24.5, 24.5),
                    7,
                    7,
                    linewidth=1.4,
                    edgecolor="red",
                    facecolor="none",
                )
            )
        right_ax.set_title(right_title)
        left_ax.axis("off")
        right_ax.axis("off")

    fig.suptitle(
        f"Original vs Backdoor samples (target={target_class}, view={view_mode})",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    clean_dataset = _load_client_trainset(args)
    poisoned_dataset = BackdoorDataset(
        clean_dataset,
        target_class=args.target_class,
        poison_ratio=args.poison_ratio,
    )

    if args.sample_mode == "poisoned":
        candidate_indices = sorted(poisoned_dataset.poisoned_indices)
    else:
        candidate_indices = list(range(len(clean_dataset)))

    indices = _choose_indices(candidate_indices, args.num_samples, args.seed)
    output_path = Path(args.output)

    _render_comparison(
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        indices=indices,
        dataset_name=args.dataset,
        target_class=args.target_class,
        output_path=output_path,
        view_mode=args.view_mode,
    )

    print(f"Saved comparison PNG: {output_path}")
    print(f"Used sample indices: {list(indices)}")
    print(f"Poisoned pool size: {len(poisoned_dataset.poisoned_indices)}")


if __name__ == "__main__":
    main()