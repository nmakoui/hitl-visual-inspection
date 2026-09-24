# Progress log

Human-in-the-Loop Visual Inspection System - build log.
Each entry follows: Done / Results / Issues / Next.

## Phase 1 - Visual similarity search and first Hugging Face demo

### 2026-09-23 - Phase 1, Task 1: Repository setup
Done: Created the project folder and Python 3.12 virtual environment;
initialised git with a .gitignore covering the venv, secrets, caches,
data and model weights; created the full folder structure from the
repository layout (configs, data, docs, notebooks, tests,
src/inspection/* subpackages, .github/workflows); added pyproject.toml
(src layout, ruff config, pytest config) and requirements.txt
(pytest, ruff, python-dotenv); installed the inspection package in
editable mode; set up .env / .env.example for secrets; wrote a first
helper function (normalise_confidence) with three passing tests;
added a README stub.
Results: ruff check . -> All checks passed. pytest -v -> 3 passed.
Issues / limitations: None yet.
Repository pushed to https://github.com/nmakoui/hitl-visual-inspection.
Next: Begin Task 2: download MVTec AD, choose 4-6 categories, write a
data loader.

### 2026-09-23/24 - Phase 1, Task 2: Category selection, dataset download, data loader
Done: Chose 5 MVTec AD categories: bottle, hazelnut, screw, capsule,
metal_nut - selected for defect-type variety (3-5 defect types each)
and a mix of object geometries, while keeping to smaller/medium-
resolution object categories rather than the largest texture
categories (carpet, wood, tile, leather). Downloaded the full MVTec
AD dataset (5.27 GB) via the Kaggle API into data/raw/mvtec_ad/.
Wrote configs/datasets.yaml recording the category list, and
src/inspection/datasets.py with list_mvtec_samples(), which lists
every image for a given category and split and attaches the matching
ground-truth mask path for defective test images. Wrote 4 unit tests
using a synthetic in-memory folder structure (pytest's tmp_path
fixture), all passing.
Results: Real image counts per category (train / test):
bottle 209/83, hazelnut 391/110, screw 320/160, capsule 219/132,
metal_nut 220/115. Total: 1,359 train images, 600 test images across
the 5 categories (1,959 total). ruff check . -> All checks passed.
pytest -v -> 7 passed.
Issues / limitations: Initial full-dataset download (5.27 GB via
Kaggle) was interrupted twice by laptop sleep/hibernate; resolved by
disabling sleep while plugged in. The other 10 MVTec AD categories
were downloaded but are unused by this project.
Next: Phase 1, Task 3 - extract DINOv2 and CLIP image embeddings for
these 1,959 images.

### 2026-09-24 - Phase 1, Task 3: DINOv2 and CLIP embedding extraction
Done: Installed torch (CPU build), torchvision, and transformers. Wrote
src/inspection/retrieval/embeddings.py with load/embed functions for
both DINOv2 (facebook/dinov2-base) and CLIP (openai/clip-vit-base-
patch32), plus extract_dataset_embeddings() which embeds every image
across the 5 chosen categories and saves results to
data/processed/embeddings/. Wrote 2 unit tests checking embedding
shapes. Ran extraction on all 1,959 images with both models.
Results: dinov2_embeddings.npy shape (1959, 768); clip_embeddings.npy
shape (1959, 512); metadata.csv has 1960 lines (1959 + header).
File sizes: dinov2 6,018,176 bytes, clip 4,012,160 bytes, metadata
138,721 bytes. Full extraction run took 42m50s on CPU. ruff check .
-> All checks passed. pytest -v -> 9 passed in 76.62s.
Issues / limitations: Hit two real bugs from using very recent
library versions (transformers 5.17.0): (1) AutoImageProcessor
required torchvision, not installed by default - fixed by adding it.
(2) CLIPModel.get_image_features() now returns a wrapped output
object instead of a raw tensor - fixed by unwrapping .image_embeds/
.pooler_output defensively. Also hit repeated download failures
(Hugging Face's newer "Xet" transfer backend failing on an unstable
connection) - fixed by setting HF_HUB_DISABLE_XET=1, now baked
directly into embeddings.py via os.environ.setdefault() so it
applies automatically in any environment, including Colab later.
Next: Phase 1, Task 4 - run PostgreSQL with pgvector in Docker, load
these embeddings, add an HNSW index and a similarity query.

### 2026-09-24 - Phase 1, Task 4-5: pgvector storage and retrieval evaluation
Done: Set up PostgreSQL with pgvector via Docker Compose
(docker-compose.yml, configs/init_db.sql). Wrote
src/inspection/retrieval/vector_store.py: creates an images table
(category, split, defect_type, image_path, dinov2_embedding vector(768),
clip_embedding vector(512)), loads Task 3's embeddings into it, adds
HNSW indexes (vector_cosine_ops) on both embedding columns, and a
find_similar() cosine-distance query with optional self-exclusion.
Wrote src/inspection/retrieval/evaluation.py: evaluate_retrieval()
computes mean precision@k and recall@k per the project's definition
of a hit (same category and defect_type), using every image in the
dataset as a query in turn. Wrote round-trip tests against the real
local Postgres for both modules.
Results: All 1,959 images loaded and indexed. Evaluation at k=5,
1,959 queries evaluated (0 skipped) for both embedding types:
DINOv2 - mean precision@5 = 0.8134, mean recall@5 = 0.0293.
CLIP - mean precision@5 = 0.7836, mean recall@5 = 0.0265.
DINOv2 outperforms CLIP on both metrics for this dataset. ruff check .
-> All checks passed. pytest -v -> 11 passed.
Issues / limitations: Recall@k is structurally very low here because
"good" (defect-free) images dominate the dataset in large groups
(e.g. hundreds of hazelnut "good" images), so even a perfect top-5
scores recall = 5/(large group size). Precision@5 is the more
informative metric for this dataset's class balance. pgvector's
Python Vector wrapper needed .to_numpy() to convert back to a plain
array - not a raw list/array as initially assumed.
Next: try CLIP's zero-shot text-to-image search as the PDF's
suggested extra, then FastAPI /similar endpoint (Task 6).

### 2026-09-24 - Phase 1, Task 5 (extra): CLIP zero-shot text-to-image search
Done: Added embed_text_clip() to embeddings.py and a small demo script
(text_search_demo.py) trying 3 plain-text queries against the indexed
clip_embedding column - a capability only CLIP has, since DINOv2 has
no text understanding.
Results: "crack in capsule" -> correctly top-matched capsule/crack.
"scratch on metal nut" -> top match was screw/manipulated_front (wrong
category and defect). "broken bottle" -> top match was metal_nut/flip
(wrong category and defect). 1 of 3 zero-shot queries hit the correct
category.
Issues / limitations: CLIP's zero-shot text search does not
generalise well to this dataset's specific industrial vocabulary
(screws, metal nuts, capsules) and defect terminology - plausibly
because these are far from the everyday photo captions CLIP was
originally trained on. This is a genuine, documented limitation, not
a bug in the retrieval code.
Next: Phase 1, Task 6 - FastAPI /similar endpoint.

### 2026-09-24 - Phase 1, Task 6: FastAPI /similar endpoint
Done: Added embed_pil_image_dinov2() and embed_text_clip() to
embeddings.py (refactored embed_image_dinov2 to reuse the PIL-based
version). Wrote src/inspection/api/main.py: a FastAPI app with /health
and /similar (upload an image, get top-k similar cases via DINOv2,
the stronger embedding per Task 5's evaluation). Model loads once at
startup via FastAPI's lifespan, not per request. Wrote 3 tests using
FastAPI's TestClient (health check, real similarity search through
the full request cycle, rejection of non-image files). Manually
verified the interactive /docs page in a browser.
Results: All 14 project tests pass. ruff check . -> All checks
passed. Real database still holds exactly 1,959 rows after the full
test suite runs.
Issues / limitations: Found and fixed two real bugs from this task:
(1) tests/test_vector_store.py and tests/test_evaluation.py were
dropping the real, shared "images" table as part of their fixtures,
breaking production data every time the full suite ran - fixed by
giving every table-touching function (create_tables, find_similar,
add_hnsw_indexes, evaluate_retrieval) an explicit table_name
parameter, defaulting to "images" for real use and overridden to
"test_images" only in test files, so tests are now fully isolated
from production data. (2) The reload script (vector_store.py's
__main__ block) only ever inserted rows with no way to clear old
data first, so re-running it after any earlier partial run left
duplicate rows (1961 instead of 1959) - fixed by adding
TRUNCATE TABLE images before every reload, making it safe to re-run
any number of times.
Next: Phase 1, Task 7 - Hugging Face Space (public demo), using
FAISS or an in-memory index since a Space cannot reach the local
Postgres database.