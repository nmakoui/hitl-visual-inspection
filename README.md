# Human-in-the-Loop Visual Inspection System

A system that inspects product images from a production or packaging line. It
detects and outlines surface defects, flags unusual items even when the
defect type has never been seen before, reads batch codes and expiry dates
from labels, and retrieves visually similar past cases. Confident decisions
are made automatically; uncertain ones go to a human reviewer, and every
reviewer decision becomes a new training label.

Author: Nastaran Makoui | github.com/nmakoui

## Project status

Phase 1 (visual similarity search and first demo) complete. See
`docs/progress.md` for the full task-by-task build log.

## Setup

Requires Python 3.11 or newer, Docker Desktop, and a Kaggle account (for
dataset download).

```bash
git clone https://github.com/nmakoui/hitl-visual-inspection.git
cd hitl-visual-inspection
python -m venv .venv
```

Activate the environment:
- Windows (cmd): `.venv\Scripts\activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Mac/Linux: `source .venv/bin/activate`

Then install:

```bash
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and fill in real values: an Anthropic API key
(used from Phase 3 onward) and PostgreSQL credentials of your choosing.

## Running the tests

```bash
ruff check .
pytest -v
```

Some tests require Docker's PostgreSQL container to be running (see below).

## Running the database

```bash
docker compose up -d
python -m inspection.retrieval.vector_store
```

The second command downloads nothing itself - it loads embeddings already
produced by `python -m inspection.retrieval.embeddings` (see Phase 1 below)
into Postgres, and is safe to re-run at any time.

## Running the API

```bash
uvicorn inspection.api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation,
including the `/similar` endpoint (upload an image, get the top-k most
similar cases).

## Running the local demo (Gradio)

```bash
python -m inspection.space.app
```

Visit the printed local URL (typically `http://127.0.0.1:7860`). This demo
is fully self-contained (a 192-image subset with its own FAISS index) and
does not require Docker or Postgres.

## Phase 1 - Visual similarity search and first demo

### What was built

- A data loader (`src/inspection/datasets.py`) for 5 MVTec AD categories
  (bottle, hazelnut, screw, capsule, metal_nut), chosen for defect-type
  variety and a mix of object geometries.
- Image embeddings from two vision foundation models, DINOv2
  (`facebook/dinov2-base`) and CLIP (`openai/clip-vit-base-patch32`), for
  all 1,959 images across the 5 categories.
- A PostgreSQL database with the pgvector extension, storing both sets of
  embeddings with HNSW indexes for fast cosine-similarity search.
- A FastAPI `/similar` endpoint: upload an image, get the top-k most
  visually similar past cases, with their category, defect type, and
  distance score.
- A CLIP zero-shot text-to-image search demo, querying the same embeddings
  with plain-text descriptions instead of an image.
- A self-contained Gradio demo, using a 192-image subset and a FAISS index,
  runnable without any database or Docker dependency.

### Results

Retrieval was evaluated with mean precision@5 and mean recall@5 across all
1,959 images (each image used as a query in turn, excluding itself). A
"hit" is a retrieved image sharing the same category and defect type as the
query.

| Model  | Mean precision@5 | Mean recall@5 |
|--------|-------------------|----------------|
| DINOv2 | 0.8134             | 0.0293         |
| CLIP   | 0.7836             | 0.0265         |

DINOv2 outperformed CLIP on both metrics for this dataset. Recall@5 is
structurally low for both models: "good" (defect-free) images dominate the
dataset in large groups (e.g. hundreds of images per category), so even a
perfect top-5 result scores a small fraction of a large relevant pool.
Precision@5 is the more informative metric given this dataset's class
balance.

CLIP's zero-shot text-to-image search - querying with a plain-text
description instead of an image, a capability DINOv2 does not have - was
tried on 3 example queries. Only 1 of 3 ("crack in capsule") returned a
correct top match; the other 2 ("scratch on metal nut", "broken bottle")
did not. CLIP's text understanding does not generalise well to this
dataset's specific industrial vocabulary, plausibly because it is
trained on general photo captions rather than inspection terminology.

Image counts by category (train / test, "train" is defect-free only):

| Category  | Train | Test |
|-----------|-------|------|
| bottle    | 209   | 83   |
| hazelnut  | 391   | 110  |
| screw     | 320   | 160  |
| capsule   | 219   | 132  |
| metal_nut | 220   | 115  |
| **Total** | 1,359 | 600  |

15 automated tests pass (`pytest -v`), covering the data loader, both
embedding functions, the vector store and its similarity search, retrieval
evaluation, the FastAPI endpoint, and the Gradio demo.

### Limitations

- Hugging Face changed its policy in 2026 so that Gradio and Docker Spaces
  now require a paid PRO subscription to create; only Static Spaces remain
  free. As a result, the Gradio demo above is not currently hosted as a
  live public Space - it is fully built and verified working locally, but
  not deployed. This is a budget decision, not a technical limitation.
- Recall@5 is not directly comparable across categories with very
  different group sizes (see Results, above).
- The other 10 MVTec AD categories were downloaded alongside the 5 used by
  this project but are not used.

## Repository structure

See `docs/progress.md` for the detailed, task-by-task build log, and the
project description for the full phased plan.