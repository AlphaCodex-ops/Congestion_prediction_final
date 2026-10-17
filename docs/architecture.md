# Architecture

## Data path

`LoadScenario` profiles produce deterministic synthetic demand or can be
connected to an external traffic generator. `NetworkConnector` is the boundary
for reading interface counters, inspecting routes, and applying link costs.
`StatsCollector` polls that boundary and flushes CSV telemetry. `FeatureBuilder`
creates time-series features and a future-congestion label. `ModelTrainer`
fits a gradient-boosted classifier; `Predictor` validates feature shape and
applies the training-only scaler. `DecisionEngine` applies hysteresis and
cooldown before `Actuator` requests any changes. `Verifier` can compare
before/after telemetry and roll back a harmful route.

## Connector boundary

The `SimulatedConnector` is safe to use on Windows and in tests. The
`MininetOspfConnector` controls FRRouting daemons through private VTY sockets
on caller-supplied Mininet router nodes. It reads interface statistics and
routes in those nodes' network namespaces, and requires explicit host,
interface, capacity, and initial-cost mappings. It does not create the
topology, manage daemon lifetimes, persist configuration, or connect the
prediction controller automatically. OVS bridges alone still do not implement
OSPF. The guided Mininet lab script can explicitly run this loop in dry-run
mode or arm lab-only actuation; actuation is disabled by default.

## Model/data safety

- CSV input must contain `utilization_pct`; `timestamp` and packet counters are
  optional.
- When a `scenario` column exists, lags and forecast labels do not cross scenario
  boundaries.
- If multiple scenario IDs exist, entire scenarios are held out together;
  without multiple scenarios, samples are split chronologically and the training
  tail is purged by the target horizon.
- `StandardScaler` is fit on training features only.
- Generated demo data is labeled synthetic and must not be presented as
  measured network performance.

## Current limitations

- The local policy comparison is a simple illustrative simulation, not a
  packet-level or routing-protocol benchmark.
- A project-owned Mininet topology launcher and automated OSPF convergence
  measurement remain to be implemented.
- The model and decision engine are not automatically wired to the real
  connector. Live lab control requires explicit caller integration; production
  access controls, authorization, and failure handling are out of scope.
- The future label uses a fixed time horizon and utilization threshold; it
  should be recalibrated using measured traffic.
