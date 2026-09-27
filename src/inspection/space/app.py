"""Gradio demo: upload a product image, find visually similar cases.

Runs standalone (no Postgres, no Docker) using a precomputed FAISS index
over a 192-image subset - suitable for a public Hugging Face Space.
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import yaml
from PIL import Image

from inspection.retrieval.embeddings import embed_pil_image_dinov2, load_dinov2

SUBSET_DIR = Path(__file__).parent / "subset"

_model, _processor = load_dinov2()
_index = faiss.read_index(str(SUBSET_DIR / "faiss_index.bin"))
_metadata = yaml.safe_load((SUBSET_DIR / "metadata.yaml").read_text())


def find_similar_cases(query_image: Image.Image, k: int) -> list[tuple[Image.Image, str]]:
    """Embed the uploaded image and return the k most similar subset images."""
    embedding = embed_pil_image_dinov2(query_image, _model, _processor)
    embedding = np.expand_dims(embedding, axis=0)

    distances, indices = _index.search(embedding, k)

    results = []
    for distance, idx in zip(distances[0], indices[0], strict=True):
        entry = _metadata[idx]
        image_path = SUBSET_DIR / "images" / entry["filename"]
        label = f"{entry['category']} / {entry['defect_type']} (distance={distance:.2f})"
        results.append((Image.open(image_path), label))

    return results


demo = gr.Interface(
    fn=find_similar_cases,
    inputs=[
        gr.Image(type="pil", label="Upload a product image"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of similar cases"),
    ],
    outputs=gr.Gallery(label="Similar past cases", columns=5),
    title="Human-in-the-Loop Visual Inspection - Similarity Search Demo",
    description=(
        "Upload a product image (bottle, hazelnut, screw, capsule, or metal nut) "
        "to find visually similar past cases from a 192-image sample, using a "
        "DINOv2 vision embedding and FAISS similarity search."
    ),
    examples=[
        [str(p), 5] for p in sorted((SUBSET_DIR / "images").glob("*.png"))[:3]
    ],
)

if __name__ == "__main__":
    demo.launch()