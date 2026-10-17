# Troubleshooting

## Python version too old

Use Python 3.10 or newer and create a fresh `.venv`. Check `python --version` to
verify the interpreter that PowerShell resolves.

## Package import or installation error

Run `.\.venv\Scripts\python.exe -m pip install -e ".[test,plots]"` from the
repository root, then `python -m congestion_project.cli doctor`.

## Training says there are too few samples

Collect a longer run or reduce `target_horizon_sec`/rolling windows. Ensure that
the CSV has numeric `utilization_pct` values and that the chronological split
still contains both a training interval and held-out samples.

## Training has only one class

The training interval must include both future-congested and future-clear
samples. Add ramp/burst/periodic experiments and inspect class counts before
model fitting.

## Accuracy looks very high but congestion is missed

Inspect recall, F1, the confusion matrix, ROC-AUC, and the majority-class
baseline. Accuracy alone is not an acceptable congestion-detection claim.

## Real-network connector says it is not configured

This is expected. Follow the routing-plane selection in
[deployment.md](deployment.md). The project deliberately does not treat OVS as
an OSPF daemon or issue unreviewed network-control commands.

## Plot command says matplotlib is missing

Install the optional chart dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
```
