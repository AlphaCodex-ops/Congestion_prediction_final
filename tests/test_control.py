import pytest

from congestion_project.control import Actuator, CongestionController, DecisionEngine, Verifier
from congestion_project.simulation import SimulatedConnector


CONFIG = {
    "threshold_high": 0.7,
    "threshold_low": 0.4,
    "cooldown_sec": 60,
    "reactive_utilization_threshold": 85,
    "revert_delay_sec": 120,
    "primary_link": ["s1", "s3"],
    "backup_link": ["s1", "s2"],
    "primary_cost": 10,
    "backup_cost": 100,
    "reroute_cost": 999,
}


def test_decision_engine_reroutes_then_reverts_after_hysteresis_and_delays():
    engine = DecisionEngine(CONFIG)
    reroute = engine.decide(0.9, 70, now=100)
    assert reroute["action"] == "reroute"
    assert reroute["new_costs"] == {"s1-s3": 999, "s1-s2": 10}
    engine.record_execution(reroute, now=100)
    assert engine.decide(0.2, 30, now=150)["action"] == "hold"
    revert = engine.decide(0.2, 30, now=221)
    assert revert["action"] == "revert"
    assert revert["new_costs"] == {"s1-s3": 10, "s1-s2": 100}


def test_decision_engine_rejects_non_finite_inputs():
    engine = DecisionEngine(CONFIG)
    with pytest.raises(ValueError, match="probability"):
        engine.decide(float("nan"), 10)


def test_reactive_fallback_and_cooldown():
    engine = DecisionEngine(CONFIG)
    assert engine.decide(0.1, 86, now=1)["action"] == "reroute"
    engine.record_execution(engine.decide(0.9, 90, now=1), now=1)
    decision = engine.decide(0.9, 90, now=10)
    assert decision["action"] == "hold"
    assert not decision["should_execute"]


def test_simulated_actuator_changes_route_and_restores_it():
    connector = SimulatedConnector()
    actuator = Actuator(connector, {"s1-s3": 10, "s1-s2": 100})
    assert actuator.verify_routes("h1", "h2", ["s1", "s3", "s2"])[0]
    actuator.set_costs({"s1-s3": 999, "s1-s2": 10})
    assert actuator.verify_routes("h1", "h2", ["s1", "s2"])[0]
    actuator.rollback()
    assert actuator.verify_routes("h1", "h2", ["s1", "s3", "s2"])[0]


def test_actuator_rejects_costs_outside_allowlist():
    actuator = Actuator(SimulatedConnector(), {"s1-s3": 10})
    with pytest.raises(ValueError, match="allowlist"):
        actuator.set_costs({"s2-s3": 10})


def test_actuator_rolls_back_partial_update():
    class FailingConnector(SimulatedConnector):
        def set_link_cost(self, src, dst, cost):
            if (src, dst, cost) == ("s1", "s2", 10):
                raise OSError("simulated route command failure")
            super().set_link_cost(src, dst, cost)

    connector = FailingConnector()
    actuator = Actuator(connector, {"s1-s3": 10, "s1-s2": 100})
    with pytest.raises(OSError, match="simulated"):
        actuator.set_costs({"s1-s3": 999, "s1-s2": 10})
    assert connector.costs[frozenset(("s1", "s3"))] == 10


def test_actuator_partial_failure_restores_previously_rerouted_costs():
    class FailingConnector(SimulatedConnector):
        fail_on_backup = False

        def set_link_cost(self, src, dst, cost):
            if self.fail_on_backup and (src, dst, cost) == ("r1", "r3", 100):
                raise OSError("simulated revert failure")
            super().set_link_cost(src, dst, cost)

    connector = FailingConnector()
    actuator = Actuator(connector, {"r1-r2": 10, "r1-r3": 100})
    actuator.set_costs({"r1-r2": 999, "r1-r3": 10})
    connector.fail_on_backup = True

    with pytest.raises(OSError, match="simulated revert"):
        actuator.set_costs({"r1-r2": 10, "r1-r3": 100})

    assert actuator.get_current_costs() == {"r1-r2": 999, "r1-r3": 10}
    assert connector.costs[frozenset(("r1", "r2"))] == 999
    assert connector.costs[frozenset(("r1", "r3"))] == 10


class ControllerConnector:
    def __init__(self):
        self.costs = {"r1-r2": 10, "r1-r3": 100}
        self.changed_costs = []

    def set_link_cost(self, src, dst, cost):
        name = f"{src}-{dst}"
        self.costs[name] = cost
        self.changed_costs.append((name, cost))

    def get_route(self, src, dst):
        if self.costs["r1-r2"] > self.costs["r1-r3"]:
            return ["r1", "r3"]
        return ["r1", "r2", "r3"]

    def read_link_counters(self, src, dst):
        raise NotImplementedError

    def close(self):
        pass


class ControllerCollector:
    def get_history(self, src, dst, lookback_sec):
        return [
            {
                "timestamp": index,
                "src_switch": src,
                "dst_switch": dst,
                "utilization_pct": 90.0,
                "packets_per_sec": 100.0,
                "dropped_delta": 0,
            }
            for index in range(30)
        ]


class ControllerFeatureBuilder:
    config = {
        "sample_interval_sec": 1,
        "lags": [1, 2, 5],
        "rolling_windows_sec": [5, 10],
    }

    def build_latest(self, frame):
        return [90.0], ["utilization_pct"]


class ControllerPredictor:
    def get_feature_names(self):
        return ["utilization_pct"]

    def predict(self, vector):
        return 0.95


def make_controller(*, actuation_enabled):
    config = {
        **CONFIG,
        "primary_link": ["r1", "r2"],
        "backup_link": ["r1", "r3"],
        "source_host": "h1",
        "destination_host": "h2",
        "primary_path": ["r1", "r2", "r3"],
        "backup_path": ["r1", "r3"],
        "verification_delay_sec": 0,
    }
    connector = ControllerConnector()
    controller = CongestionController(
        connector,
        ControllerCollector(),
        ControllerPredictor(),
        config,
        ControllerFeatureBuilder(),
        poll_interval_sec=1,
        decision_log=None,
        actuation_enabled=actuation_enabled,
    )
    return controller, connector


def test_controller_dry_run_never_changes_route_costs():
    controller, connector = make_controller(actuation_enabled=False)

    controller._process_once()

    assert connector.changed_costs == []
    assert connector.costs == {"r1-r2": 10, "r1-r3": 100}
    assert not controller.engine.is_rerouted
    assert controller.get_status()["last_action_status"] == "dry_run"


def test_controller_actuation_requires_expected_initial_route():
    controller, connector = make_controller(actuation_enabled=True)
    connector.costs["r1-r2"] = 999
    connector.costs["r1-r3"] = 10

    with pytest.raises(RuntimeError, match="initial primary route"):
        controller.start()


def test_controller_actuation_verifies_route_and_stop_restores_baseline():
    controller, connector = make_controller(actuation_enabled=True)
    controller.verifier.verify_reroute = lambda *args, **kwargs: (True, "verified")

    assert controller.actuator.verify_routes(
        "h1", "h2", ["r1", "r2", "r3"]
    )[0]
    controller._process_once()

    assert connector.get_route("h1", "h2") == ["r1", "r3"]
    assert controller.engine.is_rerouted
    assert controller.get_status()["successful_reroutes"] == 1
    assert controller.get_status()["last_action_status"] == "verified"

    controller.stop()

    assert connector.get_route("h1", "h2") == ["r1", "r2", "r3"]
    assert not controller.engine.is_rerouted


def test_controller_restores_previous_costs_when_route_verification_fails():
    controller, connector = make_controller(actuation_enabled=True)
    controller.verifier.verify_reroute = lambda *args, **kwargs: (True, "verified")
    connector.get_route = lambda src, dst: ["r1", "r2", "r3"]

    with pytest.raises(RuntimeError, match="Expected route"):
        controller._process_once()

    assert connector.costs == {"r1-r2": 10, "r1-r3": 100}
    assert controller.actuator.get_current_costs() == {
        "r1-r2": 10,
        "r1-r3": 100,
    }
    assert not controller.engine.is_rerouted
    assert controller.get_status()["last_action_status"] == "rolled_back"


def test_verifier_rejects_harmful_backup_increase_and_rolls_back():
    class Collector:
        def get_history(self, src, dst, lookback_sec):
            return [
                {"timestamp": 95, "utilization_pct": 20, "dropped_delta": 0},
                {"timestamp": 105, "utilization_pct": 60, "dropped_delta": 0},
            ]

    class Rollback:
        called = False

        def rollback(self):
            self.called = True

    actuator = Rollback()
    verifier = Verifier(Collector(), actuator)
    passed, reason = verifier.verify_reroute({"s1-s3": 999}, 100, wait_sec=0)
    assert not passed
    assert "rose by 40.0" in reason
    assert actuator.called


def test_verifier_rejects_increased_packet_drops():
    class Collector:
        def get_history(self, src, dst, lookback_sec):
            return [
                {"timestamp": 95, "utilization_pct": 20, "dropped_delta": 0},
                {"timestamp": 105, "utilization_pct": 20, "dropped_delta": 1},
            ]

    class Rollback:
        called = False

        def rollback(self):
            self.called = True

    actuator = Rollback()
    passed, reason = Verifier(Collector(), actuator).verify_reroute(
        {"s1-s3": 999}, 100, wait_sec=0
    )
    assert not passed
    assert "Packet drops increased" in reason
    assert actuator.called
