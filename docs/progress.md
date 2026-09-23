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
Next: Create the GitHub repository and push (Task 1, step 8), then
begin Task 2: download MVTec AD and write the data loader.