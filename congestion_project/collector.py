from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from congestion_project.connectors import NetworkConnector

CSV_COLUMNS = [
    "timestamp",
    "src_switch",
    "dst_switch",
    "interface",
    "bytes_sent",
    "bytes_recv",
    "packets_sent",
    "packets_recv",
    "errors",
    "dropped",
    "dropped_delta",
    "drop_rate",
    "utilization_pct",
    "queue_depth",
    "bytes_per_sec",
    "packets_per_sec",
]


class StatsCollector:
    """Poll a connector on a worker thread and append every sample to CSV."""

    def __init__(
        self,
        connector: NetworkConnector,
        links: list[tuple[str, str]],
        poll_interval_sec: float = 5,
        csv_path: str | Path = "data/raw/stats.csv",
        history_size: int = 1000,
    ) -> None:
        if poll_interval_sec <= 0 or history_size <= 0:
            raise ValueError("Poll interval and history size must be greater than zero.")
        self.connector = connector
        self.links = list(links)
        self.poll_interval_sec = float(poll_interval_sec)
        self.csv_path = Path(csv_path)
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._previous: dict[tuple[str, str], tuple[float, int, int, int]] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("The stats collector is already running.")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._stream = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_COLUMNS)
        if is_new_file:
            self._writer.writeheader()
            self._stream.flush()
        self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="stats-collector", daemon=True)
        self._thread.start()

    def poll_once(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_writer") or self._stream.closed:
            raise RuntimeError("Start the stats collector before polling.")
        now = time.time()
        rows: list[dict[str, Any]] = []
        for src, dst in self.links:
            counters = self.connector.read_link_counters(src, dst)
            key = (src, dst)
            previous = self._previous.get(key)
            bytes_per_sec = 0.0
            packets_per_sec = 0.0
            dropped_delta = 0
            packet_delta = 0
            if previous is not None:
                previous_time, previous_bytes, previous_packets, previous_dropped = previous
                elapsed = counters.timestamp - previous_time
                if elapsed > 0:
                    byte_delta = counters.bytes_sent - previous_bytes
                    packet_delta = counters.packets_sent - previous_packets
                    dropped_delta = counters.dropped - previous_dropped
                    if byte_delta < 0:
                        byte_delta += 2**32
                    if packet_delta < 0:
                        packet_delta += 2**32
                    if dropped_delta < 0:
                        dropped_delta += 2**32
                    bytes_per_sec = byte_delta / elapsed
                    packets_per_sec = packet_delta / elapsed
            self._previous[key] = (
                counters.timestamp,
                counters.bytes_sent,
                counters.packets_sent,
                counters.dropped,
            )
            row = {
                "timestamp": counters.timestamp or now,
                "src_switch": src,
                "dst_switch": dst,
                "interface": f"{src}-{dst}",
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_recv": counters.packets_recv,
                "errors": counters.errors,
                "dropped": counters.dropped,
                "dropped_delta": dropped_delta,
                "drop_rate": dropped_delta / max(1, packet_delta + dropped_delta),
                "utilization_pct": counters.utilization_pct,
                "queue_depth": counters.queue_depth,
                "bytes_per_sec": bytes_per_sec,
                "packets_per_sec": packets_per_sec,
            }
            rows.append(row)
        with self._lock:
            for row in rows:
                self._history.append(row)
                self._writer.writerow(row)
            self._stream.flush()
        return rows

    def _run(self) -> None:
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self.poll_once()
                deadline += self.poll_interval_sec
                self._stop_event.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            self._error = exc
            self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if hasattr(self, "_stream") and not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        if self._error is not None:
            raise RuntimeError("Stats collector stopped after a polling error.") from self._error

    def get_latest(self, src_switch: str, dst_switch: str) -> dict[str, Any] | None:
        with self._lock:
            for item in reversed(self._history):
                if (item["src_switch"], item["dst_switch"]) == (src_switch, dst_switch):
                    return dict(item)
        return None

    def get_history(
        self,
        src_switch: str,
        dst_switch: str,
        lookback_sec: float = 60,
    ) -> list[dict[str, Any]]:
        cutoff = time.time() - lookback_sec
        with self._lock:
            return [
                dict(item)
                for item in self._history
                if (item["src_switch"], item["dst_switch"]) == (src_switch, dst_switch)
                and item["timestamp"] >= cutoff
            ]
