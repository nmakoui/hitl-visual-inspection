from pathlib import Path

from inspection.retrieval.embeddings import (
    embed_image_clip,
    embed_image_dinov2,
    load_clip,
    load_dinov2,
)

SAMPLE_IMAGE = Path("data/raw/mvtec_ad/bottle/train/good/000.png")


def test_dinov2_embedding_shape():
    model, processor = load_dinov2()
    embedding = embed_image_dinov2(SAMPLE_IMAGE, model, processor)
    assert embedding.shape == (768,)


def test_clip_embedding_shape():
    model, processor = load_clip()
    embedding = embed_image_clip(SAMPLE_IMAGE, model, processor)
    assert embedding.shape == (512,)