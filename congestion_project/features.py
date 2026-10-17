from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureBuilder:
    """Build lagged, time-ordered congestion features without fitting on test data."""

    def __init__(self, raw_csv_path: str | Path, config: dict[str, Any] | None = None) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        self.config = config or {}
        self.feature_names: list[str] = []
        self.scaler: StandardScaler | None = None
        self._training_size = 0
        self._test_size = 0
        self.split_strategy = "not_built"
        self.training_scenarios: list[str] = []
        self.test_scenarios: list[str] = []

    def _read_frame(self) -> pd.DataFrame:
        try:
            frame = pd.read_csv(self.raw_csv_path)
        except OSError as exc:
            raise FileNotFoundError(f"Could not read telemetry CSV: {self.raw_csv_path}") from exc
        if "utilization_pct" not in frame.columns:
            raise ValueError("Telemetry CSV must contain a 'utilization_pct' column.")
        if frame.empty:
            raise ValueError("Telemetry CSV contains no samples.")
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
        frame["utilization_pct"] = pd.to_numeric(frame["utilization_pct"], errors="coerce")
        frame = frame.dropna(subset=["utilization_pct"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError("Telemetry CSV has no numeric utilization samples.")
        utilization = frame["utilization_pct"].to_numpy(dtype=float)
        if not np.isfinite(utilization).all() or np.any((utilization < 0) | (utilization > 100)):
            raise ValueError("Utilization values must be finite percentages between 0 and 100.")
        target_link = self.config.get("target_link")
        if target_link and {"src_switch", "dst_switch"}.issubset(frame.columns):
            src, dst = target_link
            selected = frame[
                ((frame["src_switch"] == src) & (frame["dst_switch"] == dst))
                | ((frame["src_switch"] == dst) & (frame["dst_switch"] == src))
            ]
            if selected.empty:
                raise ValueError(f"No samples were found for target link {src}-{dst}.")
            frame = selected.reset_index(drop=True)
        return frame

    def _feature_row(
        self,
        group: pd.DataFrame,
        index: int,
        lags: list[int],
        windows: list[int],
        interval: float,
    ) -> dict[str, float]:
        utilization = group["utilization_pct"].to_numpy(dtype=float)
        row: dict[str, float] = {"utilization_pct": float(utilization[index])}
        for lag in lags:
            row[f"utilization_pct_lag{lag}"] = float(utilization[index - lag])
        for seconds in windows:
            width = max(1, math.ceil(seconds / interval))
            values = utilization[index - width + 1 : index + 1]
            row[f"utilization_rolling_mean_{seconds}s"] = float(np.mean(values))
            row[f"utilization_rolling_max_{seconds}s"] = float(np.max(values))
            row[f"utilization_rolling_std_{seconds}s"] = float(np.std(values))
        delta = float(utilization[index] - utilization[index - 1])
        prior_delta = float(utilization[index - 1] - utilization[index - 2])
        row["utilization_delta"] = delta
        row["utilization_accel"] = delta - prior_delta
        packet_column = next(
            (name for name in ("packets_per_sec", "packets_sent") if name in group.columns),
            None,
        )
        if packet_column:
            packets = pd.to_numeric(group[packet_column], errors="coerce").fillna(0).to_numpy(dtype=float)
            if packet_column == "packets_sent":
                packets = np.diff(packets, prepend=packets[0]) / interval
                packets = np.maximum(packets, 0)
            row["packets_per_sec"] = float(packets[index])
            row["packets_per_sec_lag1"] = float(packets[index - 1])
        else:
            row["packets_per_sec"] = 0.0
            row["packets_per_sec_lag1"] = 0.0
        if "drop_rate" in group.columns:
            drops = pd.to_numeric(group["drop_rate"], errors="coerce").fillna(0).to_numpy(dtype=float)
            row["drop_rate"] = float(drops[index])
        elif "dropped_delta" in group.columns:
            dropped = pd.to_numeric(group["dropped_delta"], errors="coerce").fillna(0).to_numpy(dtype=float)
            row["drop_rate"] = float(dropped[index] / max(1.0, dropped[index] + row["packets_per_sec"]))
        elif "dropped" in group.columns:
            dropped = pd.to_numeric(group["dropped"], errors="coerce").fillna(0).to_numpy(dtype=float)
            row["drop_rate"] = float(dropped[index] / max(1.0, dropped[index] + row["packets_per_sec"]))
        else:
            row["drop_rate"] = 0.0
        return row

    def _build_raw_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[str]]:
        frame = self._read_frame()
        interval = float(self.config.get("sample_interval_sec", 5))
        horizon_sec = float(self.config.get("target_horizon_sec", 30))
        threshold = float(self.config.get("congestion_threshold_pct", 70))
        lags = sorted({int(value) for value in self.config.get("lags", [1, 2, 3, 6])})
        windows = sorted({int(value) for value in self.config.get("rolling_windows_sec", [10, 30])})
        if interval <= 0 or horizon_sec <= 0:
            raise ValueError("Feature configuration requires positive intervals and horizons.")
        if not 0.5 <= float(self.config.get("train_fraction", 0.7)) < 1:
            raise ValueError("Feature configuration requires 0.5 <= train_fraction < 1.")
        if not lags or min(lags) < 1 or any(window <= 0 for window in windows):
            raise ValueError("Feature lags and rolling windows must be positive.")
        horizon = max(1, math.ceil(horizon_sec / interval))
        max_history = max(max(lags), *(math.ceil(window / interval) for window in windows), 2)
        groups = (
            list(frame.groupby("scenario", sort=False, dropna=False))
            if "scenario" in frame.columns
            else [("all", frame)]
        )
        records: list[dict[str, float]] = []
        labels: list[int] = []
        global_positions: list[int] = []
        scenario_ids: list[str] = []
        for scenario_id, original_group in groups:
            group = original_group.reset_index(drop=True)
            global_positions_for_group = original_group.index.to_numpy()
            values = group["utilization_pct"].to_numpy(dtype=float)
            for index in range(max_history, len(group) - horizon):
                future_peak = float(np.max(values[index + 1 : index + horizon + 1]))
                records.append(self._feature_row(group, index, lags, windows, interval))
                labels.append(int(future_peak > threshold))
                global_positions.append(int(global_positions_for_group[index]))
                scenario_ids.append(str(scenario_id))
        if not records:
            raise ValueError(
                "Not enough samples to build features and future labels; collect more data or reduce the horizon/windows."
            )
        feature_frame = pd.DataFrame.from_records(records)
        self.feature_names = list(feature_frame.columns)
        raw_features = feature_frame.to_numpy(dtype=float)
        target = np.asarray(labels, dtype=np.int64)
        positions = np.asarray(global_positions, dtype=np.int64)
        return raw_features, target, positions, horizon, len(frame), scenario_ids

    def build_raw(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build unscaled features and future-congestion labels for every valid row."""
        features, target, _, _, _, _ = self._build_raw_data()
        return features, target, list(self.feature_names)

    def build_raw_with_groups(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Build unscaled features and labels, preserving each row's scenario ID."""
        features, target, _, _, _, scenario_ids = self._build_raw_data()
        return features, target, list(self.feature_names), scenario_ids

    def build(self) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
        fraction = float(self.config.get("train_fraction", 0.7))
        if not 0.5 <= fraction < 1:
            raise ValueError("Feature configuration requires 0.5 <= train_fraction < 1.")
        raw_features, target, positions, horizon, frame_length, scenario_ids = self._build_raw_data()
        scenarios = list(dict.fromkeys(scenario_ids))
        if len(scenarios) > 1:
            test_count = max(1, math.ceil(len(scenarios) * (1 - fraction)))
            self.training_scenarios = scenarios[:-test_count]
            self.test_scenarios = scenarios[-test_count:]
            training_scenarios = set(self.training_scenarios)
            train_mask = np.asarray(
                [scenario in training_scenarios for scenario in scenario_ids],
                dtype=bool,
            )
            test_mask = ~train_mask
            self.split_strategy = "scenario"
        else:
            global_boundary = int(frame_length * fraction)
            train_mask = positions + horizon < global_boundary
            test_mask = positions >= global_boundary
            self.training_scenarios = scenarios
            self.test_scenarios = scenarios
            self.split_strategy = "chronological"
        train_x = raw_features[train_mask]
        test_x = raw_features[test_mask]
        if len(train_x) < 2 or len(test_x) < 1:
            raise ValueError(
                "The train/test split is too small after separating scenarios or purging "
                "horizon-overlapping samples. Collect more samples or adjust "
                "train_fraction/target_horizon_sec."
            )
        self.scaler = StandardScaler()
        self.scaler.fit(train_x)
        self._training_size = len(train_x)
        self._test_size = len(test_x)
        scaled = np.vstack((self.scaler.transform(train_x), self.scaler.transform(test_x)))
        output_y = np.concatenate((target[train_mask], target[test_mask]))
        return scaled, output_y, self.scaler, list(self.feature_names)

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(X) != len(y) or len(X) != self._training_size + self._test_size:
            raise ValueError("X and y must be the arrays returned by this builder's build() method.")
        boundary = self._training_size
        return X[:boundary], X[boundary:], y[:boundary], y[boundary:]

    def build_latest(self, samples: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build an unscaled feature vector from ordered telemetry for online inference."""
        if "utilization_pct" not in samples:
            raise ValueError("Online telemetry must contain 'utilization_pct'.")
        interval = float(self.config.get("sample_interval_sec", 5))
        lags = sorted({int(value) for value in self.config.get("lags", [1, 2, 3, 6])})
        windows = sorted({int(value) for value in self.config.get("rolling_windows_sec", [10, 30])})
        frame = samples.sort_values("timestamp", kind="stable") if "timestamp" in samples else samples
        frame = frame.reset_index(drop=True)
        max_history = max(max(lags), *(math.ceil(window / interval) for window in windows), 2)
        index = len(frame) - 1
        if interval <= 0 or index < max_history:
            raise ValueError(f"At least {max_history + 1} ordered samples are required for online features.")
        row = self._feature_row(frame, index, lags, windows, interval)
        names = list(row)
        return np.asarray([row[name] for name in names], dtype=float), names

    def save_scaler(self, path: str | Path) -> Path:
        if self.scaler is None:
            raise RuntimeError("Call build() before saving the fitted scaler.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, destination)
        return destination

    @staticmethod
    def load_scaler(path: str | Path) -> StandardScaler:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Scaler file does not exist: {source}")
        scaler = joblib.load(source)
        if not isinstance(scaler, StandardScaler):
            raise TypeError(f"File does not contain a StandardScaler: {source}")
        return scaler
