# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A take-home freight rate prediction challenge (see `freight-rate-ml-assessment.pdf` and `readme.md`). The only code provided is the scorer, `score.py`; the model/training code is the candidate's to write. There is no git repo, test suite, or linter.

## Commands

```bash
python -m pip install -r requirements.txt   # matplotlib, numpy, pandas (scorer deps only)
python score.py --predictions validation_predictions.csv --december-predictions <completed december csv>
```

The scorer only validates file formats and renders `scorer_results/candidate_december.png` (override with `--output-dir`). It does NOT compute accuracy metrics; those are calculated externally by Spotter after submission, so hold out your own validation from the training data.

## Data layout

- `train-test.csv` (48,000 rows): labelled data; target is `posted_rate`. IDs `TR-xxxxxx`, dates start 2025-01-01.
- `validation.csv` (12,000 rows): unlabelled, same features minus `posted_rate`. IDs `TE-000001`..`TE-012000`, dates begin 2025-11-01, so it is a later time period than training. Use time-based splits when validating.
- Features: pickup/delivery city and lat/lon, `distance`, `equipment`, `weight`, `date`, `market_index`, `quote_signal`.
- `validation-predictions-template.csv`: `load_id,predicted_rate` with empty rates to fill.
- `december-chart-inputs.csv`: 31 rows, one per day of 2025-12, with `predicted_rate` empty. It has no `market_index` or `quote_signal`, so the model must produce December predictions from the date alone plus the fixed lane.

## Scorer constraints (score.py) that trip people up

- The readme refers to `data/train_test.csv`, `data/validation.csv`, etc. and to `validation_predictions.csv`, but the files here are flat in the repo root with hyphenated names (`train-test.csv`, `validation-predictions-template.csv`, `december-chart-inputs.csv`). Use the actual paths.
- Predictions file: exactly columns `load_id,predicted_rate` in that order, exactly 12,000 rows, unique IDs matching `TE-000001..TE-012000`, all rates positive and finite.
- December file: must keep the original 7 columns and order, exactly 31 daily rows, and the fixed inputs must not be altered (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb). Only `date` varies; rates must be positive.
- Deliverables per readme: GitHub repo with code and run instructions, `validation_predictions.csv`, PDF/DOCX report with the split approach and `candidate_december.png`, and a 2-3 minute Loom.
