from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from congestion_project.connectors import LinkCounters
from congestion_project.scenarios import BurstLoad, LoadScenario, PeriodicLoad, RampLoad


def make_scenarios(
    duration_sec: int = 1800,
    peak_load_mbps: float = 30,
    sample_interval_sec: int = 5,
) -> list[tuple[str, LoadScenario]]:
    common = {"duration_sec": duration_sec, "sample_interval_sec": sample_interval_sec}
    return [
        (
            "ramp",
            RampLoad(
                bandwidth_mbps=peak_load_mbps,
                ramp_in_sec=min(300, duration_sec * 0.3),
                hold_sec=min(900, duration_sec * 0.5),
                ramp_out_sec=min(300, duration_sec * 0.2),
                **common,
            ),
        ),
        (
            "burst",
            BurstLoad(
                bandwidth_mbps=peak_load_mbps,
                burst_in_sec=sample_interval_sec,
                hold_sec=duration_sec * 0.55,
                burst_out_sec=sample_interval_sec,
                **common,
            ),
        ),
        (
            "periodic",
            PeriodicLoad(
                bandwidth_mbps=peak_load_mbps,
                min_bw_mbps=peak_load_mbps * 0.1,
                max_bw_mbps=peak_load_mbps,
                period_sec=600,
                **common,
            ),
        ),
    ]


def generate_synthetic_samples(
    scenarios: Iterable[tuple[str, LoadScenario]],
    *,
    primary_capacity_mbps: float = 10,
    start_timestamp: float = 1_800_000_000.0,
) -> pd.DataFrame:
    """Create clearly labeled, deterministic telemetry for local development."""
    if primary_capacity_mbps <= 0:
        raise ValueError("Primary link capacity must be greater than zero.")
    rows: list[dict[str, Any]] = []
    next_timestamp = float(start_timestamp)
    for scenario_name, scenario in scenarios:
        count = int(scenario.duration_sec // scenario.sample_interval_sec)
        for sample_index in range(count):
            elapsed = sample_index * scenario.sample_interval_sec
            load_mbps = scenario.get_bandwidth_at_time(elapsed)
            utilization = min(100.0, load_mbps / primary_capacity_mbps * 100.0)
            packet_rate = load_mbps * 125
            drops = max(0, int((utilization - 80) * packet_rate / 10_000))
            rows.append(
                {
                    "timestamp": next_timestamp,
                    "scenario": scenario_name,
                    "src_switch": "s1",
                    "dst_switch": "s3",
                    "interface": "s1-eth2",
                    "bytes_sent": int(load_mbps * 125_000 * elapsed),
                    "bytes_recv": int(load_mbps * 125_000 * elapsed * 0.99),
                    "packets_sent": int(packet_rate * elapsed),
                    "packets_recv": int(packet_rate * elapsed) - drops,
                    "errors": 0,
                    "dropped": drops,
                    "dropped_delta": drops,
                    "drop_rate": drops / max(1, int(packet_rate * scenario.sample_interval_sec) + drops),
                    "utilization_pct": utilization,
                    "queue_depth": max(0, int((utilization - 65) * 2)),
                    "load_mbps": load_mbps,
                }
            )
            next_timestamp += scenario.sample_interval_sec
        next_timestamp += 3_600
    return pd.DataFrame.from_records(rows)


class SimulatedConnector:
    """In-memory connector used by Windows demo and tests."""

    def __init__(self) -> None:
        self.costs: dict[frozenset[str], int] = {
            frozenset(("s1", "s3")): 10,
            frozenset(("s1", "s2")): 100,
        }
        self.closed = False

    def read_link_counters(self, src_switch: str, dst_switch: str) -> LinkCounters:
        if self.closed:
            raise RuntimeError("The simulated connector is closed.")
        return LinkCounters(
            timestamp=pd.Timestamp.now().timestamp(),
            src_switch=src_switch,
            dst_switch=dst_switch,
            bytes_sent=0,
            bytes_recv=0,
            packets_sent=0,
            packets_recv=0,
        )

    def set_link_cost(self, src_switch: str, dst_switch: str, cost: int) -> None:
        if self.closed:
            raise RuntimeError("The simulated connector is closed.")
        if not 1 <= cost <= 65535:
            raise ValueError("Link cost must be between 1 and 65535.")
        self.costs[frozenset((src_switch, dst_switch))] = cost

    def get_route(self, src_host: str, dst_host: str) -> list[str]:
        if self.closed:
            raise RuntimeError("The simulated connector is closed.")
        primary_cost = self.costs[frozenset(("s1", "s3"))]
        backup_cost = self.costs[frozenset(("s1", "s2"))]
        return ["s1", "s3", "s2"] if primary_cost <= backup_cost else ["s1", "s2"]

    def close(self) -> None:
        self.closed = True


def save_samples(samples: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(destination, index=False)
    return destination


def evaluate_routing_policies(samples: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Compare capacity-pressure metrics for synthetic static/reactive/predictive policies.

    This is an illustrative simulation, not a measured Mininet/OSPF benchmark.
    """
    utilization = samples["utilization_pct"].to_numpy(dtype=float)
    reactive = utilization.copy()
    predictive = utilization.copy()
    if len(utilization):
        for index in range(1, len(utilization)):
            if utilization[index] >= 85:
                reactive[index] = min(100.0, utilization[index] * 0.65)
            if utilization[index] >= 60:
                predictive[index] = min(100.0, utilization[index] * 0.55)
            else:
                predictive[index] = utilization[index]
    policies = [
        ("static_simulated", utilization),
        ("reactive_simulated", reactive),
        ("predictive_simulated", predictive),
    ]
    rows = [
        {
            "policy": name,
            "peak_utilization_pct": float(values.max()) if len(values) else 0.0,
            "mean_utilization_pct": float(values.mean()) if len(values) else 0.0,
            "samples_over_85_pct": float(np.mean(values >= 85) * 100) if len(values) else 0.0,
        }
        for name, values in policies
    ]
    result = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def plot_routing_comparison(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot the illustrative routing-policy CSV when optional matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Routing comparison charts require matplotlib. Install with: pip install -e '.[plots]'"
        ) from exc
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ax = comparison.plot(
        x="policy",
        y=["peak_utilization_pct", "mean_utilization_pct"],
        kind="bar",
        rot=15,
        ylabel="Utilization (%)",
        title="Illustrative simulated routing policy comparison",
    )
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination
