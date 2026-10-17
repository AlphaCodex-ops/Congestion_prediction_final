# Training guide

## Run a complete local sample

```powershell
.\.venv\Scripts\python.exe -m congestion_project.cli demo --duration 1800
```

Review `data/models/metrics.json`, particularly its held-out accuracy,
precision, recall, F1, ROC-AUC, confusion matrix, and majority-class baseline.
Synthetic data is convenient for checking the pipeline but cannot establish
real-world model quality.

## Train on collected data

Your CSV must include `utilization_pct`; recommended fields are `timestamp`,
`src_switch`, `dst_switch`, `packets_per_sec` or `packets_sent`, `dropped`,
and a `scenario` identifier. Collect each independent scenario with its own
identifier so a feature window cannot bridge separate experiments.

```powershell
.\.venv\Scripts\python.exe -m congestion_project.cli train `
  --input data/raw/combined_training.csv --plots
```

To create a synthetic input explicitly:

```powershell
.\.venv\Scripts\python.exe -m congestion_project.cli generate `
  --duration 1800 --output data/raw/synthetic_training.csv
```

When there are multiple scenario IDs, the last fraction of whole scenarios is
held out; no scenario is split across training and test. If there is only one
scenario or no scenario column, the split is chronological and samples whose
labels cross the split boundary are omitted. In either case, the scaler is fit
only on training samples. Check `metrics.json` for the split strategy and the
train/test scenario lists. Inspect class balance and the majority-class
baseline; high accuracy can be misleading when congestion is rare.

The saved model bundle and scaler are `data/models/model_v1.joblib` and
`data/models/scaler_v1.joblib`. Keep them together with the feature configuration
and use the `Predictor` API for inference.

## Validate a saved model on a separate capture

Use `validate` to score an existing model against an independent CSV. This
command does not train or modify the model; it builds the same unscaled
features and future-congestion labels, then applies the scaler saved with the
model. Use the same feature configuration and target link used during training.

```bash
python -m congestion_project.cli validate \
  --config config.mininet.yaml \
  --input data/raw/mininet_training_run2.csv \
  --model data/models/mininet_run1/model_v1.joblib \
  --scaler data/models/mininet_run1/scaler_v1.joblib
```

The result includes class counts, accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, and majority-class baseline, plus per-scenario metrics and
confusion matrices when the CSV has a `scenario` column. A one-class validation
file cannot provide ROC-AUC. To inspect a different probability cutoff, pass
`--probability-threshold`, for example:

```bash
python -m congestion_project.cli validate \
  --config config.mininet.yaml \
  --input data/raw/mininet_training_run4.csv \
  --model data/models/mininet_development_grouped_run1-3/model_v1.joblib \
  --scaler data/models/mininet_development_grouped_run1-3/scaler_v1.joblib \
  --probability-threshold 0.75
```

The default remains 0.5. Pick thresholds using development data, then reserve
an untouched capture for evaluating a fixed threshold if you need an
independent performance claim. Run4 has already informed model development, so
it is not an untouched final test. A run5 capture is optional for a
prototype/demo, but needed for that stronger claim. This offline command does
not connect the model to live routing.
