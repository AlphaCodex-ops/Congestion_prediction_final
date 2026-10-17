# Offline project status and handoff

## Completed in the isolated Mininet lab

- FRRouting OSPF adjacencies, host reachability, and primary/backup route
  changes were exercised in the caller-owned Mininet topology.
- The project connector read real interface counters and changed/restored
  explicitly mapped OSPF costs.
- Traffic shaping was applied to `r1-eth1` for capture experiments and was
  removed afterward (`noqueue` was confirmed).
- Separate captures were recorded as run1 through run4. Run1 through run3
  were combined for a development model; run4 was kept out of that training
  input and used for offline validation.

## Model evidence and limitations

After adding scenario-held-out splitting, a separate grouped-split model was
trained from run1 through run3 and scored on run4 at the default 0.5 probability
cutoff:

- 982 labeled samples: 246 clear and 736 congested.
- Confusion matrix: `[[159, 87], [0, 736]]`.
- Accuracy 0.911, precision 0.894, recall 1.000, F1 0.944, ROC-AUC 1.000.
- All false positives were in the below-threshold traffic runs, averaging
  about 68.3% utilization. The near-threshold-low runs averaged about 73.4%
  and were detected.

Run4 has already been examined during development, including a cutoff sweep.
Treat it as development evidence, not an untouched final test. A run5 capture is
optional for a prototype or course demonstration; it is needed only if the
project requires an independent evaluation after fixing the model and cutoff.

The current captures mostly use steady offered loads. They do not establish
forecast performance on dynamic ramps or bursts, performance across different
topologies, or real-world routing benefit.

## Reproduce the offline model workflow

From the Ubuntu project directory, the user's captured-data workflow is:

```bash
cd /home/johan/work/Congestion_prediction

./.venv/bin/python3 -m congestion_project.cli train \
  --config config.mininet.yaml \
  --input data/raw/mininet_development_run1-3.csv \
  --output-dir data/models/mininet_development_grouped_run1-3

./.venv/bin/python3 -m congestion_project.cli validate \
  --config config.mininet.yaml \
  --input data/raw/mininet_training_run4.csv \
  --model data/models/mininet_development_grouped_run1-3/model_v1.joblib \
  --scaler data/models/mininet_development_grouped_run1-3/scaler_v1.joblib
```

Use `--probability-threshold` to inspect alternative cutoffs offline. Do not
describe a cutoff chosen by inspecting run4 as independently validated.

For the software-only demo and regression checks:

```bash
./.venv/bin/python3 -m pytest -q
./.venv/bin/python3 -m congestion_project.cli demo --duration 600
```

The demo's routing-policy comparison is illustrative simulation output, not a
Mininet benchmark.

## Guarded live-lab controller integration

- The caller-owned lab script now wires the saved model, `StatsCollector`,
  `CongestionController`, and `MininetOspfConnector` together.
- Controller mode is off by default. `dry-run` records candidate actions
  without changing OSPF costs. `actuate` requires an explicit lab-only
  confirmation flag and the measured 2 Mbps shaper.
- Actuation checks the starting primary route, verifies the resulting path,
  uses the existing congestion/drop verifier, restores the previous costs if
  path verification fails, and restores baseline OSPF costs when the lab exits.
- Live-controller telemetry and decisions are written to separate CSV files;
  the long training capture is opt-in and is skipped by default.
- Controller modes now serve a loopback-only live dashboard in the lab process.
  It renders measured telemetry and the observed route, and its traffic buttons
  control a bounded h3-to-h2 UDP load from a host attached to r2, alongside a
  low-rate managed h1-to-h2 flow. The controller measures the shared r2-r3
  bottleneck and the dashboard shows both observed routes.

## Remaining for a full empirical evaluation

- No project-owned Mininet/FRR topology launcher was added; the lab script
  remains caller-owned.
- No static/reactive/predictive policy comparison has been measured under
  identical Mininet traffic traces.
- The run4 cutoff sweep informed development; no untouched evaluation of a
  fixed alternative probability cutoff is available. Run5 is optional for a
  prototype/demo, but needed for that independent claim.
- Dynamic ramp/burst traffic and alternative topology performance remain
  unmeasured.
- No production network support is claimed.

Keep actuation limited to the disposable Mininet topology. Do not point the lab
connector or these flags at production routers.
