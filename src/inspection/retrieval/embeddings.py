"""Extract DINOv2 and CLIP image embeddings using Hugging Face transformers."""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet backend is unreliable on flaky connections

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

DINOV2_MODEL_NAME = "facebook/dinov2-base"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def load_dinov2():
    """Load the DINOv2 model and its image processor, ready for CPU inference."""
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
    model = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    model.eval()
    return model, processor


def load_clip():
    """Load the CLIP model and its image processor, ready for CPU inference."""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def embed_pil_image_dinov2(image: Image.Image, model, processor) -> np.ndarray:
    """Return a single DINOv2 embedding vector (768-d) for an already-loaded image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # The [CLS] token (position 0) is DINOv2's standard image-level summary vector.
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def embed_image_dinov2(image_path: Path, model, processor) -> np.ndarray:
    """Return a single DINOv2 embedding vector (768-d) for one image on disk."""
    image = Image.open(image_path).convert("RGB")
    return embed_pil_image_dinov2(image, model, processor)


def embed_image_clip(image_path: Path, model, processor) -> np.ndarray:
    """Return a single CLIP image embedding vector (512-d) for one image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Newer transformers versions may wrap the result in an output object
    # instead of returning the tensor directly - unwrap it if so.
    if isinstance(outputs, torch.Tensor):
        embedding = outputs
    elif hasattr(outputs, "image_embeds"):
        embedding = outputs.image_embeds
    elif hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        raise TypeError(f"Unexpected output type from get_image_features: {type(outputs)}")

    return embedding.squeeze().numpy()
def extract_dataset_embeddings(
    root: Path, categories: list[str], output_dir: Path
) -> None:
    """Embed every image in the given categories with DINOv2 and CLIP, saving
    the results to disk: one metadata CSV plus two .npy embedding arrays,
    row-aligned with the metadata.
    """
    import csv

    from tqdm import tqdm

    from inspection.datasets import list_mvtec_samples

    output_dir.mkdir(parents=True, exist_ok=True)

    dinov2_model, dinov2_processor = load_dinov2()
    clip_model, clip_processor = load_clip()

    all_samples = []
    for category in categories:
        for split in ("train", "test"):
            all_samples.extend(list_mvtec_samples(root, category, split))

    dinov2_vectors = np.zeros((len(all_samples), 768), dtype=np.float32)
    clip_vectors = np.zeros((len(all_samples), 512), dtype=np.float32)

    with open(output_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "category", "split", "defect_type", "image_path"])

        for i, sample in enumerate(tqdm(all_samples, desc="Embedding images")):
            dinov2_vectors[i] = embed_image_dinov2(
                sample.image_path, dinov2_model, dinov2_processor
            )
            clip_vectors[i] = embed_image_clip(
                sample.image_path, clip_model, clip_processor
            )
            writer.writerow(
                [i, sample.category, sample.split, sample.defect_type, str(sample.image_path)]
            )

    np.save(output_dir / "dinov2_embeddings.npy", dinov2_vectors)
    np.save(output_dir / "clip_embeddings.npy", clip_vectors)


if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(Path("configs/datasets.yaml").read_text())
    extract_dataset_embeddings(
        root=Path("data/raw/mvtec_ad"),
        categories=config["mvtec_categories"],
        output_dir=Path("data/processed/embeddings"),
    )
    print("Done. Saved to data/processed/embeddings/")

def embed_text_clip(text: str, model, processor) -> np.ndarray:
    """Return a CLIP text embedding vector (512-d) for a plain-text query."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    if isinstance(outputs, torch.Tensor):
        embedding = outputs
    elif hasattr(outputs, "text_embeds"):
        embedding = outputs.text_embeds
    elif hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        raise TypeError(f"Unexpected output type from get_text_features: {type(outputs)}")

    return embedding.squeeze().numpy()