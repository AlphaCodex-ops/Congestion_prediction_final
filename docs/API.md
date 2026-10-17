# Component API

Public classes are in `congestion_project`:

- `connectors.NetworkConnector`: protocol with `read_link_counters`,
  `set_link_cost`, `get_route`, and `close`.
- `connectors.LinkCounters`: normalized interface-counter record.
- `connectors.MininetOspfConnector`: FRRouting connector for caller-owned
  Mininet router nodes. Construction requires explicit router, VTY socket,
  host, interface, capacity, and current-cost mappings. It reads Linux
  interface counters in each router namespace, traces IPv4 routes, and applies
  verified OSPF costs to both ends of an allowlisted link. `close()` disables
  the adapter; the topology owner remains responsible for stopping Mininet and
  FRR.
- `simulation.SimulatedConnector`: local implementation of route/cost methods.
- `simulation.plot_routing_comparison(frame, path)`: optional plot of illustrative
  policy-comparison CSV outputs.
- `collector.StatsCollector`: `start()`, `stop()`, `poll_once()`,
  `get_latest(src, dst)`, and `get_history(src, dst, lookback_sec)`.
- `scenarios.LoadScenario`, `RampLoad`, `BurstLoad`, `PeriodicLoad`, and
  `FailureScenario`: `get_bandwidth_at_time(t_sec)` returns a target Mbps value.
- `features.FeatureBuilder`: `build()` returns `(X, y, scaler, feature_names)`;
  `split(X, y)` returns training/test arrays, holding out whole scenarios when
  multiple scenario IDs are present and using a purged chronological split
  otherwise; `build_latest(frame)` returns an unscaled vector and ordered names
  for inference.
- `training.ModelTrainer`: `train()`, `evaluate()`, `save_model(path)`, and
  optional `plot_results(directory)`.
- `training.Predictor`: `predict(vector)`, `predict_batch(matrix)`, and
  `get_feature_names()`.
- `control.DecisionEngine`: `decide(probability, utilization, now=None)`,
  `record_execution(decision)`, `get_state()`, and `log_decision(decision)`.
- `control.Actuator`: `set_costs(mapping)`, `verify_routes(...)`, and
  `rollback()`.
- `control.CongestionController`: background `start()`, `stop()`, and
  `get_status()`. Route actuation is disabled by default; enabling it requires
  expected primary/backup paths and host endpoints in the controller config.

Configuration loads from `config.yaml` with `config.load_config(path)`.
Unknown connector operations fail explicitly; there is no success-shaped
fallback.
