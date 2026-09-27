"""Build a small image subset + FAISS index for the public Hugging Face Space demo."""

import shutil
from pathlib import Path

import faiss
import numpy as np
import yaml

from inspection.datasets import list_mvtec_samples

IMAGES_PER_CATEGORY = 40


def select_subset_samples(root: Path, categories: list[str]) -> list:
    """Pick a mix of good/defective test images, evenly spread across categories."""
    selected = []
    for category in categories:
        test_samples = list_mvtec_samples(root, category, "test")

        by_defect_type: dict[str, list] = {}
        for sample in test_samples:
            by_defect_type.setdefault(sample.defect_type, []).append(sample)

        defect_types = sorted(by_defect_type)
        per_type = max(1, IMAGES_PER_CATEGORY // len(defect_types))

        category_selection = []
        for defect_type in defect_types:
            category_selection.extend(by_defect_type[defect_type][:per_type])

        selected.extend(category_selection[:IMAGES_PER_CATEGORY])

    return selected


def build_subset(
    root: Path, categories: list[str], output_dir: Path, dinov2_model, dinov2_processor
) -> None:
    """Copy selected images, embed them, and save a FAISS index + metadata."""
    from inspection.retrieval.embeddings import embed_image_dinov2

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = select_subset_samples(root, categories)
    print(f"Selected {len(samples)} images for the subset.")

    embeddings = np.zeros((len(samples), 768), dtype=np.float32)
    metadata = []

    for i, sample in enumerate(samples):
        new_filename = f"{i:04d}_{sample.category}_{sample.defect_type}.png"
        shutil.copy(sample.image_path, images_dir / new_filename)

        embeddings[i] = embed_image_dinov2(sample.image_path, dinov2_model, dinov2_processor)

        metadata.append(
            {
                "filename": new_filename,
                "category": sample.category,
                "defect_type": sample.defect_type,
            }
        )

    index = faiss.IndexFlatL2(768)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss_index.bin"))

    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    print(f"Saved {len(samples)} images, FAISS index, and metadata to {output_dir}")


if __name__ == "__main__":
    from inspection.retrieval.embeddings import load_dinov2

    config = yaml.safe_load(Path("configs/datasets.yaml").read_text())
    model, processor = load_dinov2()

    build_subset(
        root=Path("data/raw/mvtec_ad"),
        categories=config["mvtec_categories"],
        output_dir=Path("src/inspection/space/subset"),
        dinov2_model=model,
        dinov2_processor=processor,
    )