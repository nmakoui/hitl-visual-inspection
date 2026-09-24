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

### 2026-09-23 - Phase 1, Task 2: Category selection
Done: Chose 5 MVTec AD categories to use for this project: bottle,
hazelnut, screw, capsule, metal_nut. Chosen for defect-type variety
(3-5 defect types each) and to cover different object geometries
(bottles, natural/organic hazelnut texture, small elongated screws,
capsules, rotatable metal nuts), while keeping to smaller/medium-
resolution object categories rather than the largest texture
categories (carpet, wood, tile, leather).

### 2026-09-24 - Phase 1, Task 2: Dataset download and category selection
Done: Downloaded the full MVTec AD dataset (5.27 GB) via the Kaggle
API into data/raw/mvtec_ad/. Chose 5 categories to use for this
project: bottle, hazelnut, screw, capsule, metal_nut - selected for
defect-type variety (3-5 defect types each) and a mix of object
geometries, while keeping to smaller/medium-resolution object
categories rather than the largest texture categories (carpet, wood,
tile, leather). Wrote configs/datasets.yaml recording the category
list, and src/inspection/datasets.py with list_mvtec_samples(), which
lists every image for a given category and split and attaches the
matching ground-truth mask path for defective test images. Wrote
4 unit tests using a synthetic in-memory folder structure (pytest's
tmp_path fixture), all passing.
Results: Real image counts per category (train / test):
bottle 209/83, hazelnut 391/110, screw 320/160, capsule 219/132,
metal_nut 220/115. Total: 1,359 train images, 600 test images across
the 5 categories. ruff check . -> All checks passed. pytest -v ->
7 passed.
Issues / limitations: Initial full-dataset download attempts were
interrupted twice by laptop sleep/hibernate; resolved by disabling
sleep while plugged in. The other 10 MVTec AD categories were
downloaded but are unused by this project.
Next: Phase 1, Task 3 - extract DINOv2 and CLIP image embeddings for
these 1,959 images.
Issues / limitations: Initial full-dataset download (5.27 GB via
Kaggle) was interrupted twice by laptop sleep; restarted with sleep
disabled.
Next: confirm download completed, then write the config listing
these categories and the data loader.
