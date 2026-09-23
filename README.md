# Human-in-the-Loop Visual Inspection System

A system that inspects product images from a production or packaging line. It
detects and outlines surface defects, flags unusual items even when the
defect type has never been seen before, reads batch codes and expiry dates
from labels, and retrieves visually similar past cases. Confident decisions
are made automatically; uncertain ones go to a human reviewer, and every
reviewer decision becomes a new training label.

Author: Nastaran Makoui | github.com/nmakoui

## Project status

Phase 1, Task 1 (repository setup) in progress. See `docs/progress.md` for
the detailed log.

## Setup

Requires Python 3.11 or newer.

```bash
git clone <repo-url>
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

Copy `.env.example` to `.env` and fill in real values (see that file for
which keys are needed).

## Running the tests

```bash
ruff check .
pytest -v
```

## Repository structure

See `docs/progress.md` for the current build log, and the project
description for the full phased plan.