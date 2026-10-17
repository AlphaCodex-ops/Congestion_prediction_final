from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from congestion_project.config import load_config
from congestion_project.control import DecisionEngine
from congestion_project.features import FeatureBuilder
from congestion_project.simulation import (
    evaluate_routing_policies,
    generate_synthetic_samples,
    make_scenarios,
    plot_routing_comparison,
    save_samples,
)
from congestion_project.training import ModelTrainer, Predictor

ROOT = Path(__file__).resolve().parents[1]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _configuration(path: str) -> dict[str, Any]:
    return load_config(Path(path))


def _feature_configuration(config: dict[str, Any]) -> dict[str, Any]:
    feature_config = dict(config.get("feature_builder", {}))
    feature_config["target_link"] = config.get("decision_engine", {}).get(
        "primary_link", ("s1", "s3")
    )
    return feature_config


def _generate(config: dict[str, Any], duration_sec: int | None = None):
    simulation = config.get("simulation", {})
    scenario_config = config.get("simulation_scenarios", {})
    duration = duration_sec or int(scenario_config.get("duration_sec", 1800))
    interval = int(simulation.get("sample_interval_sec", 5))
    scenarios = make_scenarios(
        duration,
        float(scenario_config.get("peak_load_mbps", 30)),
        interval,
    )
    return generate_synthetic_samples(
        scenarios,
        primary_capacity_mbps=float(simulation.get("primary_capacity_mbps", 10)),
    )


def _train(samples_path: Path, output_dir: Path, config: dict[str, Any], make_plots: bool) -> dict[str, Any]:
    feature_config = _feature_configuration(config)
    builder = FeatureBuilder(samples_path, feature_config)
    X, y, _, feature_names = builder.build()
    X_train, X_test, y_train, y_test = builder.split(X, y)
    trainer = ModelTrainer(
        X_train,
        y_train,
        X_test,
        y_test,
        config.get("trainer", {}),
        feature_names,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir / "model_v1.joblib")
    builder.save_scaler(output_dir / "scaler_v1.joblib")
    _write_json(
        {
            "feature_names": feature_names,
            "feature_config": feature_config,
            "split_strategy": builder.split_strategy,
            "train_scenarios": builder.training_scenarios,
            "test_scenarios": builder.test_scenarios,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "metrics": trainer.metrics,
        },
        output_dir / "metrics.json",
    )
    if make_plots:
        trainer.plot_results(output_dir.parent / "results")
    return {
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "feature_count": len(feature_names),
        "split_strategy": builder.split_strategy,
        "train_scenario_count": len(builder.training_scenarios),
        "test_scenario_count": len(builder.test_scenarios),
        "metrics": trainer.metrics,
        "model_path": str(output_dir / "model_v1.joblib"),
        "scaler_path": str(output_dir / "scaler_v1.joblib"),
    }


def command_doctor(_: argparse.Namespace) -> int:
    print(f"Python: {platform.python_version()} ({platform.system()})")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required.")
        return 1
    dependencies = ("numpy", "pandas", "sklearn", "yaml", "joblib")
    missing = []
    for module in dependencies:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print(f"ERROR: Missing required Python packages: {', '.join(missing)}")
        return 1
    print("Core Python dependencies: available")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Optional plots: unavailable (install with pip install -e '.[plots]')")
    else:
        print("Optional plots: available")
    print("Windows/local simulator workflow: available")
    print("Live Mininet/OSPF: not configured; see docs/deployment.md")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config = _configuration(args.config)
    samples = _generate(config, args.duration)
    destination = save_samples(samples, args.output)
    print(f"Wrote {len(samples)} synthetic telemetry rows to {destination}")
    print("These generated samples are for software development, not real network measurements.")
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _configuration(args.config)
    result = _train(Path(args.input), Path(args.output_dir), config, args.plots)
    print(json.dumps(result, indent=2, default=_json_default))
    if result["metrics"]["accuracy"] < result["metrics"]["baseline_accuracy"]:
        print("WARNING: model accuracy is below the majority-class baseline.")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = _configuration(args.config)
    probability_threshold = float(getattr(args, "probability_threshold", 0.5))
    if not np.isfinite(probability_threshold) or not 0 <= probability_threshold <= 1:
        raise ValueError("Probability threshold must be between 0 and 1.")
    builder = FeatureBuilder(Path(args.input), _feature_configuration(config))
    features, labels, feature_names, scenario_ids = builder.build_raw_with_groups()
    predictor = Predictor(args.model, args.scaler)
    if feature_names != predictor.get_feature_names():
        raise ValueError(
            "Validation feature names or order do not match the saved model. "
            "Use the same feature configuration that was used for training."
        )
    probabilities = predictor.predict_batch(features)
    predictions = (probabilities >= probability_threshold).astype(int)
    scenario_metrics = {}
    for scenario_id in dict.fromkeys(scenario_ids):
        scenario_mask = np.asarray(scenario_ids) == scenario_id
        scenario_labels = labels[scenario_mask]
        scenario_predictions = predictions[scenario_mask]
        scenario_metrics[scenario_id] = {
            "samples": int(len(scenario_labels)),
            "class_counts": {
                "clear": int(np.count_nonzero(scenario_labels == 0)),
                "congested": int(np.count_nonzero(scenario_labels == 1)),
            },
            "accuracy": float(accuracy_score(scenario_labels, scenario_predictions)),
            "precision": float(
                precision_score(scenario_labels, scenario_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(scenario_labels, scenario_predictions, zero_division=0)
            ),
            "f1": float(f1_score(scenario_labels, scenario_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                scenario_labels, scenario_predictions, labels=[0, 1]
            ),
        }
    result = {
        "input": str(args.input),
        "model_path": str(args.model),
        "probability_threshold": probability_threshold,
        "samples": int(len(labels)),
        "class_counts": {
            "clear": int(np.count_nonzero(labels == 0)),
            "congested": int(np.count_nonzero(labels == 1)),
        },
        "scenario_metrics": scenario_metrics,
        "metrics": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(labels, probabilities))
                if len(np.unique(labels)) == 2
                else None
            ),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]),
            "baseline_accuracy": float(max(np.mean(labels == 0), np.mean(labels == 1))),
        },
    }
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import pandas as pd

    samples = pd.read_csv(args.input)
    result = evaluate_routing_policies(samples, args.output)
    print(result.to_string(index=False))
    if args.plots:
        chart_path = Path(args.output).with_name("routing_comparison.png")
        plot_routing_comparison(result, chart_path)
        print(f"Comparison chart: {chart_path}")
    print("Note: policy values are illustrative simulator outputs, not Mininet measurements.")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    config = _configuration(args.config)
    output_root = Path(args.output_dir)
    samples = _generate(config, args.duration)
    raw_path = save_samples(samples, output_root / "raw" / "synthetic_training.csv")
    trained = _train(raw_path, output_root / "models", config, args.plots)
    policies = evaluate_routing_policies(
        samples, output_root / "results" / "routing_comparison.csv"
    )
    if args.plots:
        plot_routing_comparison(policies, output_root / "results" / "routing_comparison.png")
    decision_config = dict(config.get("decision_engine", {}))
    engine = DecisionEngine(decision_config, output_root / "results" / "decisions.csv")
    decision = engine.decide(prediction_prob=0.95, current_utilization_pct=75, now=1000)
    engine.log_decision(decision)
    if decision["should_execute"]:
        engine.record_execution(decision, now=1000)
    _write_json(decision, output_root / "results" / "demo_decision.json")
    print(f"Generated synthetic samples: {len(samples)} ({raw_path})")
    print(
        f"Trained model: {trained['train_samples']} train / "
        f"{trained['test_samples']} test samples; "
        f"accuracy={trained['metrics']['accuracy']:.3f}, "
        f"baseline={trained['metrics']['baseline_accuracy']:.3f}"
    )
    print(f"Routing policy comparison: {output_root / 'results' / 'routing_comparison.csv'}")
    print(f"Predictive decision example: {decision['action']} - {decision['reason']}")
    print(policies.to_string(index=False))
    print("All demo data is synthetic. Live routing is intentionally not attempted.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="congestion-project",
        description="Train and evaluate a congestion predictor using synthetic or recorded telemetry.",
    )
    parser.add_argument("--version", action="version", version="congestion-project 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="Check local Python prerequisites.")
    doctor.set_defaults(handler=command_doctor)
    generate = subparsers.add_parser("generate", help="Generate synthetic training telemetry.")
    generate.add_argument("--config", default=str(ROOT / "config.yaml"))
    generate.add_argument("--duration", type=int)
    generate.add_argument("--output", default=str(ROOT / "data/raw/synthetic_training.csv"))
    generate.set_defaults(handler=command_generate)
    train = subparsers.add_parser("train", help="Train on an existing telemetry CSV.")
    train.add_argument("--config", default=str(ROOT / "config.yaml"))
    train.add_argument("--input", required=True)
    train.add_argument("--output-dir", default=str(ROOT / "data/models"))
    train.add_argument("--plots", action="store_true")
    train.set_defaults(handler=command_train)
    validate = subparsers.add_parser(
        "validate",
        help="Score a saved model on a separate telemetry CSV without retraining.",
    )
    validate.add_argument("--config", default=str(ROOT / "config.yaml"))
    validate.add_argument("--input", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--scaler", required=True)
    validate.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for reporting classification metrics (default: 0.5).",
    )
    validate.set_defaults(handler=command_validate)
    evaluate = subparsers.add_parser("evaluate", help="Generate the simulated policy comparison.")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", default=str(ROOT / "data/results/routing_comparison.csv"))
    evaluate.add_argument("--plots", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)
    demo = subparsers.add_parser("demo", help="Run synthetic data, training, and routing-policy demo.")
    demo.add_argument("--config", default=str(ROOT / "config.yaml"))
    demo.add_argument("--duration", type=int)
    demo.add_argument("--output-dir", default=str(ROOT / "data"))
    demo.add_argument("--plots", action="store_true")
    demo.set_defaults(handler=command_demo)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
