# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Solution

LightGBM on `log(posted_rate)` with time-based (expanding-window) validation. See `outputs/report.docx` for the approach, validation results and the December chart.

Run from the repo root (the data files are flat in the root with hyphenated names):

```bash
python -m pip install -r requirements.txt
python -m src.validate     # time-based CV -> outputs/cv_results.csv, outputs/cv_summary.csv
python -m src.predict      # -> validation_predictions.csv, outputs/december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
python -m src.report       # -> outputs/report.docx
```

Layout: `src/data.py` (loading, cleaning, label-outlier flagging), `src/features.py`, `src/validate.py` (CV and baselines), `src/predict.py` (final fit and both prediction files), `src/report.py`.
