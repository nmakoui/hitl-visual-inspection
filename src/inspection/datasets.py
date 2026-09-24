"""Locate and list MVTec AD images on disk, by category and split."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MVTecSample:
    """One image from MVTec AD, with its label and (if any) mask path."""

    category: str
    split: str  # "train" or "test"
    defect_type: str  # "good", or a defect name such as "scratch"
    image_path: Path
    mask_path: Path | None  # None for "good" images - they have no mask


def list_mvtec_samples(root: Path, category: str, split: str) -> list[MVTecSample]:
    """List every image for one category and split ("train" or "test").

    For defective test images, attaches the matching ground-truth mask
    path when one exists on disk. "good" images never have a mask.
    """
    split_dir = root / category / split
    samples: list[MVTecSample] = []

    for defect_dir in sorted(split_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name

        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path = None
            if defect_type != "good":
                candidate = (
                    root / category / "ground_truth" / defect_type
                    / f"{image_path.stem}_mask.png"
                )
                if candidate.exists():
                    mask_path = candidate

            samples.append(
                MVTecSample(category, split, defect_type, image_path, mask_path)
            )

    return samples