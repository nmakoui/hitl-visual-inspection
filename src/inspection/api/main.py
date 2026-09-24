"""FastAPI app exposing a /similar image-similarity endpoint."""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from inspection.retrieval.embeddings import embed_pil_image_dinov2, load_dinov2
from inspection.retrieval.vector_store import find_similar, get_connection

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the DINOv2 model once at startup, not on every request."""
    model, processor = load_dinov2()
    _models["model"] = model
    _models["processor"] = processor
    yield
    _models.clear()


app = FastAPI(title="HITL Visual Inspection - Similarity Search", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/similar")
async def similar(file: UploadFile = File(...), k: int = 5) -> dict:
    """Upload an image, get the top-k most similar images from the dataset."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    embedding = embed_pil_image_dinov2(image, _models["model"], _models["processor"])

    conn = get_connection()
    try:
        results = find_similar(conn, embedding, "dinov2_embedding", k=k)
    finally:
        conn.close()

    return {"query_filename": file.filename, "results": results}