from argparse import Namespace
import json

import pandas as pd
import pytest

from congestion_project.cli import command_demo, command_train, command_validate
from congestion_project.connectors import MininetOspfConnector
from congestion_project.control import Actuator


class FakeMininetNode:
    def __init__(self, name, state):
        self.name = name
        self.state = state
        self.counters = {
            "tx_bytes": 0,
            "rx_bytes": 0,
            "tx_packets": 0,
            "rx_packets": 0,
        }

    def pexec(self, *args):
        if args[0] == "ip" and args[1:4] == ("-json", "-s", "link"):
            stats = {
                "tx": {
                    "bytes": self.counters["tx_bytes"],
                    "packets": self.counters["tx_packets"],
                    "errors": 0,
                    "dropped": 0,
                },
                "rx": {
                    "bytes": self.counters["rx_bytes"],
                    "packets": self.counters["rx_packets"],
                    "errors": 0,
                    "dropped": 0,
                },
            }
            return json.dumps([{"stats64": stats}]), "", 0
        if args[0] == "ip" and args[1:3] == ("-json", "route"):
            if self.name == "r1":
                interface = (
                    "r1-eth2"
                    if self.state["costs"][("r1", "r1-eth2")] < 20
                    else "r1-eth1"
                )
            else:
                interface = "r2-eth1"
            return json.dumps([{"dev": interface}]), "", 0
        if args[0] == "vtysh":
            commands = [
                args[index + 1]
                for index, value in enumerate(args[:-1])
                if value == "-c"
            ]
            if commands[0].startswith("show ip ospf interface "):
                interface = commands[0].rsplit(" ", maxsplit=1)[1]
                cost = self.state["costs"][(self.name, interface)]
                if self.state.get("reject_cost") == (self.name, interface, cost):
                    cost += 1
                return f"  Cost: {cost}\n", "", 0
            interface = None
            cost = None
            for command in commands:
                if command.startswith("interface "):
                    interface = command.split(maxsplit=1)[1]
                elif command.startswith("ip ospf cost "):
                    cost = int(command.rsplit(" ", maxsplit=1)[1])
            if interface is not None and cost is not None:
                self.state["costs"][(self.name, interface)] = cost
            return "", "", 0
        return "", f"Unexpected command: {args!r}", 1


def make_mininet_connector():
    state = {
        "costs": {
            ("r1", "r1-eth2"): 100,
            ("r3", "r3-eth1"): 100,
            ("r1", "r1-eth1"): 10,
            ("r2", "r2-eth0"): 10,
            ("r2", "r2-eth1"): 10,
            ("r3", "r3-eth0"): 10,
        }
    }
    routers = {
        name: FakeMininetNode(name, state)
        for name in ("r1", "r2", "r3")
    }
    connector = MininetOspfConnector(
        routers=routers,
        link_interfaces={
            ("r1", "r3"): {"r1": "r1-eth2", "r3": "r3-eth1"},
            ("r1", "r2"): {"r1": "r1-eth1", "r2": "r2-eth0"},
            ("r2", "r3"): {"r2": "r2-eth1", "r3": "r3-eth0"},
        },
        vty_sockets={name: f"/tmp/{name}/vty" for name in routers},
        host_endpoints={
            "h1": ("r1", "192.168.1.2"),
            "h2": ("r3", "192.168.3.2"),
        },
        link_capacities_mbps={
            ("r1", "r3"): 10,
            ("r1", "r2"): 10,
            ("r2", "r3"): 10,
        },
        initial_costs={
            ("r1", "r3"): 100,
            ("r1", "r2"): 10,
            ("r2", "r3"): 10,
        },
    )
    return connector, routers, state


def test_mininet_ospf_connector_reads_counters_and_calculates_utilization(monkeypatch):
    connector, routers, _ = make_mininet_connector()
    ticks = iter((10.0, 11.0))
    monkeypatch.setattr("congestion_project.connectors.time.time", lambda: next(ticks))
    routers["r1"].counters["tx_bytes"] = 1_000
    first = connector.read_link_counters("r1", "r3")
    routers["r1"].counters["tx_bytes"] = 126_000
    second = connector.read_link_counters("r1", "r3")

    assert first.utilization_pct == 0
    assert second.bytes_sent == 126_000
    assert second.utilization_pct == pytest.approx(10)


def test_mininet_ospf_connector_changes_route_and_actuator_restores_cost():
    connector, _, state = make_mininet_connector()
    actuator = Actuator(connector, {"r1-r3": 100})

    assert actuator.verify_routes("h1", "h2", ["r1", "r2", "r3"])[0]
    actuator.set_costs({"r1-r3": 5})
    assert actuator.verify_routes("h1", "h2", ["r1", "r3"])[0]
    assert state["costs"][("r1", "r1-eth2")] == 5
    assert state["costs"][("r3", "r3-eth1")] == 5

    actuator.rollback()
    assert actuator.verify_routes("h1", "h2", ["r1", "r2", "r3"])[0]
    assert state["costs"][("r1", "r1-eth2")] == 100
    assert state["costs"][("r3", "r3-eth1")] == 100
    connector.close()


def test_mininet_ospf_connector_rejects_unmapped_links():
    connector, _, _ = make_mininet_connector()
    with pytest.raises(ValueError, match="not configured"):
        connector.read_link_counters("r1", "r9")


def test_mininet_ospf_connector_rolls_back_both_endpoints_on_verification_failure():
    connector, _, state = make_mininet_connector()
    state["reject_cost"] = ("r3", "r3-eth1", 5)

    with pytest.raises(RuntimeError, match="did not confirm"):
        connector.set_link_cost("r1", "r3", 5)

    assert state["costs"][("r1", "r1-eth2")] == 100
    assert state["costs"][("r3", "r3-eth1")] == 100


def test_demo_runs_end_to_end_on_local_synthetic_data(tmp_path):
    result = command_demo(
        Namespace(
            config="config.yaml",
            duration=600,
            output_dir=str(tmp_path / "data"),
            plots=False,
        )
    )
    assert result == 0
    assert (tmp_path / "data/raw/synthetic_training.csv").is_file()
    assert (tmp_path / "data/models/model_v1.joblib").is_file()
    assert (tmp_path / "data/models/metrics.json").is_file()
    assert (tmp_path / "data/results/routing_comparison.csv").is_file()


def test_validate_scores_saved_model_on_external_capture(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
feature_builder:
  sample_interval_sec: 1
  lags: [1, 2]
  rolling_windows_sec: [2]
  target_horizon_sec: 3
  congestion_threshold_pct: 70
  train_fraction: 0.5
trainer:
  n_estimators: 20
  random_state: 7
decision_engine:
  primary_link: [r1, r2]
""",
        encoding="utf-8",
    )

    def write_runs(path, scenarios):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": scenario_index * 100 + index,
                    "scenario": name,
                    "src_switch": "r1",
                    "dst_switch": "r2",
                    "utilization_pct": value,
                    "packets_per_sec": 10,
                }
                for scenario_index, (name, value) in enumerate(scenarios)
                for index in range(80)
            ]
        )
        frame.to_csv(path, index=False)

    training_csv = tmp_path / "training.csv"
    external_csv = tmp_path / "external.csv"
    write_runs(
        training_csv,
        [
            ("train-clear-1", 20.0),
            ("train-congested-1", 95.0),
            ("train-clear-2", 25.0),
            ("train-congested-2", 90.0),
        ],
    )
    write_runs(external_csv, [("holdout-clear", 30.0), ("holdout-congested", 90.0)])
    model_dir = tmp_path / "model"
    assert command_train(
        Namespace(
            config=str(config_path),
            input=str(training_csv),
            output_dir=str(model_dir),
            plots=False,
        )
    ) == 0
    capsys.readouterr()

    assert command_validate(
        Namespace(
            config=str(config_path),
            input=str(external_csv),
            model=str(model_dir / "model_v1.joblib"),
            scaler=str(model_dir / "scaler_v1.joblib"),
            probability_threshold=0.5,
        )
    ) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["probability_threshold"] == 0.5
    assert result["samples"] == 150
    assert result["class_counts"] == {"clear": 75, "congested": 75}
    assert result["metrics"]["confusion_matrix"] == [[75, 0], [0, 75]]
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["roc_auc"] == 1.0
    assert result["scenario_metrics"]["holdout-clear"]["confusion_matrix"] == [
        [75, 0],
        [0, 0],
    ]
    assert result["scenario_metrics"]["holdout-congested"]["confusion_matrix"] == [
        [0, 0],
        [0, 75],
    ]


def test_validate_rejects_probability_threshold_out_of_range(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("feature_builder: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Probability threshold"):
        command_validate(
            Namespace(
                config=str(config_path),
                input=str(tmp_path / "unused.csv"),
                model=str(tmp_path / "unused-model.joblib"),
                scaler=str(tmp_path / "unused-scaler.joblib"),
                probability_threshold=1.1,
            )
        )
