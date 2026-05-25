import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_path = repo_root / "data" / "datasets" / "NEU-DET" / "NEU-DET" / "train" / "images"
    output_path = repo_root / "paper" / "figures" / "neudet_example.png"

    fig, axes = plt.subplots(nrows=len(CLASSES), ncols=2, figsize=(6, 12))

    for i, cls in enumerate(CLASSES):
        folder_path = base_path / cls

        if folder_path.exists():
            img_files = [
                f
                for f in sorted(os.listdir(folder_path))
                if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg", ".tif", ".tiff"))
            ]

            if not img_files:
                print(f"Warning: No images found in {folder_path}")
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")
                continue

            img_path = folder_path / img_files[0]

            img = Image.open(img_path).convert("L")
            img = img.resize((32, 32))
            img_array = np.array(img)

            poisoned_array = img_array.copy()
            poisoned_array[-7:, -7:] = 255

            axes[i, 0].imshow(img_array, cmap="gray")
            axes[i, 0].set_title(f"Original | {cls}", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(poisoned_array, cmap="gray")
            axes[i, 1].set_title("Poisoned | Target=0 | Trigger ON", fontsize=10)
            axes[i, 1].axis("off")
        else:
            print(f"Warning: Could not find folder {folder_path}")
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"Image saved successfully to {output_path}")


if __name__ == "__main__":
    main()
