from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path
from typing import Any

from congestion_project.connectors import NetworkConnector


def _link_name(link: list[str] | tuple[str, str]) -> str:
    return f"{link[0]}-{link[1]}"


class DecisionEngine:
    def __init__(self, config: dict[str, Any], log_path: str | Path | None = None) -> None:
        self.config = config
        self.threshold_high = float(config.get("threshold_high", 0.7))
        self.threshold_low = float(config.get("threshold_low", 0.4))
        self.cooldown_sec = float(config.get("cooldown_sec", 60))
        self.reactive_threshold = float(config.get("reactive_utilization_threshold", 85))
        self.revert_delay_sec = float(config.get("revert_delay_sec", 120))
        if not 0 <= self.threshold_low < self.threshold_high <= 1:
            raise ValueError("Decision thresholds must satisfy 0 <= threshold_low < threshold_high <= 1.")
        if min(self.cooldown_sec, self.revert_delay_sec) < 0:
            raise ValueError("Cooldown and revert delay cannot be negative.")
        self.primary_link = tuple(config.get("primary_link", ("s1", "s3")))
        self.backup_link = tuple(config.get("backup_link", ("s1", "s2")))
        self.primary_cost = int(config.get("primary_cost", 10))
        self.backup_cost = int(config.get("backup_cost", 100))
        self.reroute_cost = int(config.get("reroute_cost", 999))
        self.last_change_time: float | None = None
        self.last_action: str | None = None
        self.is_rerouted = False
        self._reroute_time: float | None = None
        self.log_path = Path(log_path) if log_path else None

    def decide(
        self,
        prediction_prob: float,
        current_utilization_pct: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(prediction_prob) or not 0 <= prediction_prob <= 1:
            raise ValueError("Prediction probability must be between 0 and 1.")
        if not math.isfinite(current_utilization_pct) or not 0 <= current_utilization_pct <= 100:
            raise ValueError("Utilization must be between 0 and 100 percent.")
        current_time = time.time() if now is None else float(now)
        cooldown_ready = (
            self.last_change_time is None
            or current_time - self.last_change_time >= self.cooldown_sec
        )
        action = "hold"
        reason = "No routing change is currently warranted."
        costs: dict[str, int] = {}
        if self.is_rerouted:
            revert_ready = (
                self._reroute_time is not None
                and current_time - self._reroute_time >= self.revert_delay_sec
            )
            if prediction_prob <= self.threshold_low and revert_ready and cooldown_ready:
                action, reason = "revert", "Congestion risk is below the hysteresis threshold."
                costs = {
                    _link_name(self.primary_link): self.primary_cost,
                    _link_name(self.backup_link): self.backup_cost,
                }
            elif prediction_prob <= self.threshold_low and not revert_ready:
                reason = "Below the revert threshold, but the minimum reroute hold time has not elapsed."
            elif not cooldown_ready:
                reason = "A previous route change is still in its cooldown period."
        else:
            if prediction_prob >= self.threshold_high:
                action, reason = "reroute", "Predicted congestion exceeds the high hysteresis threshold."
            elif current_utilization_pct >= self.reactive_threshold:
                action, reason = "reroute", "Reactive utilization safeguard threshold was reached."
            if action == "reroute":
                if cooldown_ready:
                    costs = {
                        _link_name(self.primary_link): self.reroute_cost,
                        _link_name(self.backup_link): self.primary_cost,
                    }
                else:
                    action, reason = "hold", "A previous route change is still in its cooldown period."
        return {
            "action": action,
            "reason": reason,
            "new_costs": costs,
            "should_execute": bool(costs),
            "prediction_prob": float(prediction_prob),
            "utilization_pct": float(current_utilization_pct),
            "timestamp": current_time,
        }

    def record_execution(self, decision: dict[str, Any], now: float | None = None) -> None:
        if not decision.get("should_execute"):
            raise ValueError("Only an executable decision can be recorded.")
        if decision.get("action") not in {"reroute", "revert"}:
            raise ValueError("Only reroute and revert actions can be recorded.")
        current_time = time.time() if now is None else float(now)
        self.last_change_time = current_time
        self.last_action = decision["action"]
        self.is_rerouted = decision["action"] == "reroute"
        self._reroute_time = current_time if self.is_rerouted else None

    def record_rollback(self, now: float | None = None) -> None:
        self.last_change_time = time.time() if now is None else float(now)
        self.last_action = "rollback"
        self.is_rerouted = False
        self._reroute_time = None

    def get_state(self, now: float | None = None) -> dict[str, Any]:
        current_time = time.time() if now is None else float(now)
        return {
            "last_change_time": self.last_change_time,
            "last_action": self.last_action,
            "is_rerouted": self.is_rerouted,
            "time_since_last_change": (
                None if self.last_change_time is None else max(0.0, current_time - self.last_change_time)
            ),
        }

    def log_decision(self, decision: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", newline="", encoding="utf-8") as stream:
            fields = [
                "timestamp",
                "action",
                "reason",
                "prediction_prob",
                "utilization_pct",
                "should_execute",
                "new_costs",
                "verification",
                "actuation_status",
                "actual_route",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow({key: decision.get(key) for key in fields})


class Actuator:
    """Apply route costs using the configured connector, rolling back partial changes."""

    def __init__(self, connector: NetworkConnector, original_costs: dict[str, int]) -> None:
        self.connector = connector
        self.original_costs = dict(original_costs)
        self._current_costs = dict(original_costs)

    @staticmethod
    def _parse_link(name: str) -> tuple[str, str]:
        parts = name.split("-", maxsplit=1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid link name {name!r}; expected 'src-dst'.")
        return parts[0], parts[1]

    def set_costs(self, costs_dict: dict[str, int]) -> bool:
        applied: list[str] = []
        previous_costs = dict(self._current_costs)
        try:
            for link, cost in costs_dict.items():
                if isinstance(cost, bool) or not isinstance(cost, int) or not 1 <= cost <= 65535:
                    raise ValueError(f"Link cost must be between 1 and 65535: {cost}")
                if link not in self.original_costs:
                    raise ValueError(f"Link {link!r} is not in the configured actuator allowlist.")
                src, dst = self._parse_link(link)
                self.connector.set_link_cost(src, dst, int(cost))
                self._current_costs[link] = int(cost)
                applied.append(link)
        except Exception as exc:
            rollback_errors: list[Exception] = []
            for link in reversed(applied):
                try:
                    src, dst = self._parse_link(link)
                    previous_cost = previous_costs[link]
                    self.connector.set_link_cost(src, dst, previous_cost)
                    self._current_costs[link] = previous_cost
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    f"Cost update failed and {len(rollback_errors)} rollback operation(s) also failed."
                ) from exc
            raise
        return True

    def get_current_costs(self) -> dict[str, int]:
        return dict(self._current_costs)

    def verify_routes(
        self, src_host: str, dst_host: str, expected_path: list[str] | None = None
    ) -> tuple[bool, list[str]]:
        actual = self.connector.get_route(src_host, dst_host)
        return (expected_path is None or actual == expected_path), actual

    def rollback(self) -> None:
        self.set_costs(self.original_costs)


class Verifier:
    def __init__(
        self,
        collector: Any,
        actuator: Actuator,
        *,
        max_backup_utilization_increase_pct: float = 30,
    ) -> None:
        self.collector = collector
        self.actuator = actuator
        self.max_backup_utilization_increase_pct = max_backup_utilization_increase_pct

    def verify_reroute(
        self,
        action_taken: dict[str, int],
        time_before_reroute: float,
        *,
        wait_sec: float = 10,
        backup_link: tuple[str, str] = ("s1", "s2"),
    ) -> tuple[bool, str]:
        if not action_taken:
            raise ValueError("Reroute verification requires a non-empty action.")
        verification_window = max(float(wait_sec), 5.0)
        if wait_sec > 0:
            time.sleep(wait_sec)
        history = self.collector.get_history(*backup_link, lookback_sec=verification_window + 60)
        before = [
            row
            for row in history
            if time_before_reroute - verification_window
            <= row["timestamp"]
            <= time_before_reroute
        ]
        after = [
            row
            for row in history
            if time_before_reroute < row["timestamp"] <= time_before_reroute + verification_window
        ]
        if not before or not after:
            self.actuator.rollback()
            return False, "Insufficient before/after backup-link telemetry to verify the reroute."
        increase = float(after[-1]["utilization_pct"]) - float(before[-1]["utilization_pct"])
        if increase > self.max_backup_utilization_increase_pct:
            self.actuator.rollback()
            return False, f"Backup utilization rose by {increase:.1f} percentage points; route rolled back."
        before_drops = sum(int(row.get("dropped_delta", 0)) for row in before)
        after_drops = sum(int(row.get("dropped_delta", 0)) for row in after)
        if after_drops > before_drops:
            self.actuator.rollback()
            return False, f"Packet drops increased from {before_drops} to {after_drops}; route rolled back."
        return True, "Reroute verification passed."


class CongestionController:
    """Background prediction loop with lab actuation disabled unless explicitly enabled."""

    def __init__(
        self,
        connector: NetworkConnector,
        collector: Any,
        predictor: Any,
        config: dict[str, Any],
        feature_builder: Any,
        *,
        poll_interval_sec: float = 5,
        decision_log: str | Path = "data/results/decisions.csv",
        actuation_enabled: bool = False,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError("Controller poll interval must be positive.")
        self.connector = connector
        self.collector = collector
        self.predictor = predictor
        self.feature_builder = feature_builder
        self.poll_interval_sec = poll_interval_sec
        self.verification_delay_sec = float(config.get("verification_delay_sec", 10))
        if self.verification_delay_sec < 0:
            raise ValueError("Verification delay cannot be negative.")
        self.engine = DecisionEngine(config, decision_log)
        self.actuation_enabled = bool(actuation_enabled)
        self.source_host = config.get("source_host")
        self.destination_host = config.get("destination_host")
        self.primary_path = config.get("primary_path")
        self.backup_path = config.get("backup_path")
        if self.actuation_enabled:
            if not self.source_host or not self.destination_host:
                raise ValueError("Enabled actuation requires source_host and destination_host.")
            if not isinstance(self.primary_path, list) or not isinstance(self.backup_path, list):
                raise ValueError("Enabled actuation requires primary_path and backup_path lists.")
            if len(self.primary_path) < 2 or len(self.backup_path) < 2:
                raise ValueError("Expected primary and backup paths must each contain at least two routers.")
            if self.primary_path[0] != self.backup_path[0] or self.primary_path[-1] != self.backup_path[-1]:
                raise ValueError("Primary and backup paths must have the same source and destination routers.")
        self.actuator = Actuator(
            connector,
            {
                _link_name(tuple(config.get("primary_link", ("s1", "s3")))): int(
                    config.get("primary_cost", 10)
                ),
                _link_name(tuple(config.get("backup_link", ("s1", "s2")))): int(
                    config.get("backup_cost", 100)
                ),
            },
        )
        self.verifier = Verifier(collector, self.actuator)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._last_prediction: float | None = None
        self._last_utilization: float | None = None
        self._decisions = 0
        self._successful_reroutes = 0
        self._rollbacks = 0
        self._last_action_status: str | None = None
        self._last_decision: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Controller is already running.")
        if self.actuation_enabled:
            matches, actual = self.actuator.verify_routes(
                self.source_host,
                self.destination_host,
                self.primary_path,
            )
            if not matches:
                raise RuntimeError(
                    f"Actuation requires the initial primary route {self.primary_path!r}; "
                    f"found {actual!r}."
                )
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="congestion-controller", daemon=True)
        self._thread.start()

    def _minimum_history_size(self) -> int:
        config = getattr(self.feature_builder, "config", {})
        interval = float(config.get("sample_interval_sec", self.poll_interval_sec))
        if interval <= 0:
            raise ValueError("Feature sample interval must be positive.")
        lags = [int(value) for value in config.get("lags", [1, 2, 3, 6])]
        windows = [int(value) for value in config.get("rolling_windows_sec", [10, 30])]
        history = max(max(lags), *(math.ceil(window / interval) for window in windows), 2)
        feature_history = history + 1
        verification_window = max(self.verification_delay_sec, 5.0)
        verification_history = math.ceil(
            2 * verification_window / self.poll_interval_sec
        ) + 2
        return max(feature_history, verification_history)

    def _process_once(self) -> None:
        primary_link = tuple(self.engine.primary_link)
        history = self.collector.get_history(*primary_link, lookback_sec=300)
        if len(history) < self._minimum_history_size():
            self._last_action_status = "warming_up"
            return
        import pandas as pd

        vector, names = self.feature_builder.build_latest(pd.DataFrame(history))
        if names != self.predictor.get_feature_names():
            raise ValueError("Online feature order does not match the trained model.")
        latest = history[-1]
        probability = self.predictor.predict(vector)
        decision = self.engine.decide(probability, float(latest["utilization_pct"]))
        self._last_prediction = probability
        self._last_utilization = float(latest["utilization_pct"])
        self._decisions += 1
        decision["actuation_status"] = "enabled" if self.actuation_enabled else "dry_run"
        self._last_action_status = decision["actuation_status"]
        if decision["should_execute"]:
            if self.actuation_enabled:
                try:
                    self._execute_decision(decision)
                except Exception as exc:
                    decision["actuation_status"] = "failed"
                    decision["verification"] = f"Actuation failed: {exc}"
                    self._last_decision = dict(decision)
                    self.engine.log_decision(decision)
                    raise
            else:
                decision["verification"] = "Dry-run only; no OSPF costs were changed."
        self.engine.log_decision(decision)
        self._last_decision = dict(decision)

    def _execute_decision(self, decision: dict[str, Any]) -> None:
        previous_costs = self.actuator.get_current_costs()
        self.actuator.set_costs(decision["new_costs"])
        if decision["action"] == "reroute":
            try:
                verified, reason = self.verifier.verify_reroute(
                    decision["new_costs"],
                    decision["timestamp"],
                    wait_sec=self.verification_delay_sec,
                    backup_link=self.engine.backup_link,
                )
            except Exception:
                self.actuator.set_costs(previous_costs)
                raise
            decision["verification"] = reason
            if not verified:
                self.engine.record_rollback()
                self._rollbacks += 1
                self._last_action_status = "rolled_back"
                decision["actuation_status"] = self._last_action_status
                return
            expected_path = self.backup_path
        else:
            if self.verification_delay_sec:
                time.sleep(self.verification_delay_sec)
            expected_path = self.primary_path

        matches, actual_path = self.actuator.verify_routes(
            self.source_host,
            self.destination_host,
            expected_path,
        )
        decision["actual_route"] = actual_path
        if not matches:
            self.actuator.set_costs(previous_costs)
            if decision["action"] == "reroute":
                self.engine.record_rollback()
            self._rollbacks += 1
            self._last_action_status = "rolled_back"
            decision["verification"] = (
                f"Expected route {expected_path!r}, observed {actual_path!r}; "
                "restored the previous OSPF costs."
            )
            raise RuntimeError(decision["verification"])

        self.engine.record_execution(decision)
        if decision["action"] == "reroute":
            self._successful_reroutes += 1
        self._last_action_status = "verified"
        decision["actuation_status"] = self._last_action_status

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._process_once()
                self._stop_event.wait(self.poll_interval_sec)
        except Exception as exc:
            self._error = exc
            self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self.actuation_enabled and self.engine.is_rerouted:
            self.actuator.rollback()
            self.engine.record_rollback()
        if self._error is not None:
            raise RuntimeError("Controller stopped after an execution error.") from self._error

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "actuation_enabled": self.actuation_enabled,
            "last_action_status": self._last_action_status,
            "last_prediction": self._last_prediction,
            "last_utilization": self._last_utilization,
            "is_rerouted": self.engine.is_rerouted,
            "decisions_made": self._decisions,
            "successful_reroutes": self._successful_reroutes,
            "rollbacks": self._rollbacks,
            "error": None if self._error is None else str(self._error),
            "last_decision": (
                None
                if self._last_decision is None
                else dict(self._last_decision)
            ),
        }
