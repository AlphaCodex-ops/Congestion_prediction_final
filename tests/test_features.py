import numpy as np
import pandas as pd
import pytest

from congestion_project.features import FeatureBuilder


def _write_samples(path, count=120):
    utilization = [20.0 if index % 24 < 8 else 95.0 for index in range(count)]
    pd.DataFrame(
        {
            "timestamp": np.arange(count, dtype=float),
            "scenario": ["run-a"] * count,
            "src_switch": ["s1"] * count,
            "dst_switch": ["s3"] * count,
            "utilization_pct": utilization,
            "packets_per_sec": np.arange(count, dtype=float) + 10,
            "dropped": np.zeros(count, dtype=int),
        }
    ).to_csv(path, index=False)


def test_features_split_chronologically_and_fit_scaler_on_train_only(tmp_path):
    source = tmp_path / "samples.csv"
    _write_samples(source)
    config = {
        "sample_interval_sec": 1,
        "lags": [1, 2, 3],
        "rolling_windows_sec": [2, 4],
        "target_horizon_sec": 3,
        "congestion_threshold_pct": 70,
        "train_fraction": 0.7,
        "target_link": ["s1", "s3"],
    }
    builder = FeatureBuilder(source, config)
    X, y, scaler, names = builder.build()
    X_train, X_test, y_train, y_test = builder.split(X, y)
    assert len(X_train) and len(X_test)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-10)
    assert names[0] == "utilization_pct"
    assert "utilization_pct_lag3" in names
    assert "utilization_rolling_max_4s" in names
    assert len(y_train) + len(y_test) == len(y)
    assert scaler.n_features_in_ == len(names)


def test_scenario_groups_do_not_leak_future_labels_between_runs(tmp_path):
    source = tmp_path / "scenarios.csv"
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(60, dtype=float),
            "scenario": ["clear"] * 30 + ["congested"] * 30,
            "utilization_pct": [0.0] * 30 + [100.0] * 30,
        }
    )
    frame.to_csv(source, index=False)
    builder = FeatureBuilder(
        source,
        {
            "sample_interval_sec": 1,
            "lags": [1, 2],
            "rolling_windows_sec": [2],
            "target_horizon_sec": 2,
            "congestion_threshold_pct": 70,
            "train_fraction": 0.7,
        },
    )
    X, labels, _, _ = builder.build()
    _, _, train_labels, test_labels = builder.split(X, labels)
    assert builder.split_strategy == "scenario"
    assert builder.training_scenarios == ["clear"]
    assert builder.test_scenarios == ["congested"]
    assert np.all(train_labels == 0)
    assert np.all(test_labels == 1)
    assert labels.sum() > 0


def test_build_raw_returns_unscaled_features_and_labels_for_all_valid_samples(tmp_path):
    source = tmp_path / "samples.csv"
    _write_samples(source, count=120)
    builder = FeatureBuilder(
        source,
        {
            "sample_interval_sec": 1,
            "lags": [1, 2, 3],
            "rolling_windows_sec": [2, 4],
            "target_horizon_sec": 3,
            "congestion_threshold_pct": 70,
        },
    )

    features, labels, names = builder.build_raw()

    assert features.shape == (113, len(names))
    assert set(np.unique(labels)) == {0, 1}
    assert features[:, names.index("utilization_pct")].max() == 95.0
    assert builder.scaler is None


def test_build_raw_with_groups_preserves_scenario_for_each_feature_row(tmp_path):
    source = tmp_path / "scenarios.csv"
    pd.DataFrame(
        {
            "timestamp": np.arange(80, dtype=float),
            "scenario": ["clear"] * 40 + ["congested"] * 40,
            "utilization_pct": [20.0] * 40 + [95.0] * 40,
        }
    ).to_csv(source, index=False)
    builder = FeatureBuilder(
        source,
        {
            "sample_interval_sec": 1,
            "lags": [1, 2],
            "rolling_windows_sec": [2],
            "target_horizon_sec": 3,
            "congestion_threshold_pct": 70,
        },
    )

    features, labels, _, scenario_ids = builder.build_raw_with_groups()

    assert len(features) == len(labels) == len(scenario_ids) == 70
    assert scenario_ids == ["clear"] * 35 + ["congested"] * 35
    assert np.all(labels[:35] == 0)
    assert np.all(labels[35:] == 1)


def test_builder_reports_missing_utilization_column(tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [1, 2]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="utilization_pct"):
        FeatureBuilder(source).build()
