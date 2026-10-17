# Network Congestion Prediction

A Python prototype for predicting short-term network congestion and safely
requesting a route change. It has a fully runnable **Windows-friendly synthetic
demo**, an offline model-training path for CSV telemetry, controller logic, and
a clearly isolated connector boundary for real networks.

> **Scope:** the synthetic demo and its policy-comparison numbers are for
> software development only. The FRR connector and guarded prediction
> controller are exercised only in an isolated Mininet lab, including a
> loopback-only dashboard for controlled traffic experiments. No
> production-network support or measured real-world routing benefit is claimed.

## Start on Windows

Requirements: Windows 10/11, Python 3.10+, PowerShell, and internet access for
the first package installation. No Linux, Mininet, Docker, or network
administrator privileges are needed to run the local demo.

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m congestion_project.cli demo
```

Or create the virtual environment manually:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,plots]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m congestion_project.cli doctor
.\.venv\Scripts\python.exe -m congestion_project.cli demo
```

The demo writes sample telemetry, a trained scikit-learn model, its scaler,
metrics, a routing-policy CSV, and a sample safe decision beneath `data/`.
Confusion-matrix, feature-importance, ROC, and policy-comparison charts are
generated only when `--plots` is supplied.

## Commands

```powershell
python -m congestion_project.cli doctor
python -m congestion_project.cli generate --duration 1800 --output data/raw/training.csv
python -m congestion_project.cli train --input data/raw/training.csv --plots
python -m congestion_project.cli evaluate --input data/raw/training.csv --plots
python -m congestion_project.cli demo
```

The installed `congestion-project` command is also available after setup.
Options and defaults are shown by adding `--help`.

## What is implemented

- Deterministic ramp, burst, periodic, and failure load profiles.
- Synthetic telemetry generation for local development and tests.
- Append-only CSV telemetry collector and in-memory sample history.
- Time-ordered lag, rolling, trend, packet-rate, drop-rate, and future-congestion features.
- Scenario-held-out train/test split when multiple scenario IDs are available,
  with a purged chronological split as the single-scenario fallback; the scaler
  is fitted on training data only.
- Class-balanced gradient-boosting training, test metrics, model persistence,
  prediction wrapper, and optional confusion-matrix/feature-importance plots.
- Hysteresis, cooldown, reactive fallback, revert delay, action logging,
  transactional actuator rollback, and a controller loop.
- `NetworkConnector` interface and a simulated connector.
- Explicitly configured Mininet/FRRouting OSPF connector; live control is
  limited to the Mininet topology objects and interfaces supplied by its owner.
- Guarded prediction-controller integration in the guided Mininet lab script:
  dry-run by default, explicit lab-only actuation flags, route verification,
  rollback, and baseline restoration on shutdown.
- Loopback-only live browser dashboard for the guided lab, with measured
  topology/link state, controller decisions, and a bounded external traffic
  rate control for a host attached to the r2 LAN.
- Offline validation of a saved model against a separate telemetry CSV, with
  overall and per-scenario metrics.

## Routing design

The original PRD asks Open vSwitch/OpenFlow switches to change OSPF link costs.
Those are different control planes: OVS bridges do not run OSPF. Before live
routing is attempted, choose and validate one:

1. Run FRRouting (or another OSPF daemon) on Linux routers. The project now
   includes a Mininet/FRR connector for explicitly mapped lab topologies; or
2. Keep OVS and implement path changes through an SDN controller/OpenFlow rules,
   not OSPF costs. This alternative is not implemented.

The Mininet/FRRouting adapter can control an isolated lab topology after the
caller provides its router objects, host addresses, interface mappings, link
capacities, and current OSPF costs. It does not start the topology or
automatically connect the trained predictor to live routing by default. The caller-owned guided lab
script offers an explicit dry-run mode and a separately confirmed
lab-only-actuation mode. Read [deployment guide](docs/deployment.md) before
using either.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m congestion_project.cli demo --duration 600
```

See [architecture](docs/architecture.md), [training](docs/training_guide.md),
[API](docs/API.md), [deployment](docs/deployment.md), and
[troubleshooting](docs/troubleshooting.md). Follow the guided remaining steps
in [docs/next_steps.md](docs/next_steps.md).
