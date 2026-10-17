import csv
import time

import pytest

from congestion_project.collector import StatsCollector
from congestion_project.connectors import LinkCounters


class CounterConnector:
    def __init__(self):
        self.sample = 0
        self.fail = False

    def read_link_counters(self, src_switch, dst_switch):
        if self.fail:
            raise OSError("simulated telemetry failure")
        self.sample += 1
        return LinkCounters(
            timestamp=time.time(),
            src_switch=src_switch,
            dst_switch=dst_switch,
            bytes_sent=1000 * self.sample,
            bytes_recv=900 * self.sample,
            packets_sent=10 * self.sample,
            packets_recv=9 * self.sample,
            utilization_pct=20.0,
        )

    def set_link_cost(self, src_switch, dst_switch, cost):
        pass

    def get_route(self, src_host, dst_host):
        return ["s1", "s3", "s2"]

    def close(self):
        pass


def test_collector_flushes_csv_and_retains_history(tmp_path):
    path = tmp_path / "samples.csv"
    collector = StatsCollector(CounterConnector(), [("s1", "s3")], 0.01, path)
    collector.start()
    time.sleep(0.035)
    collector.stop()
    latest = collector.get_latest("s1", "s3")
    assert latest is not None
    assert latest["utilization_pct"] == 20
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 2
    assert float(rows[-1]["bytes_per_sec"]) > 0


def test_collector_surfaces_worker_failures(tmp_path):
    connector = CounterConnector()
    connector.fail = True
    collector = StatsCollector(connector, [("s1", "s3")], 0.01, tmp_path / "bad.csv")
    collector.start()
    time.sleep(0.03)
    with pytest.raises(RuntimeError, match="polling error"):
        collector.stop()


def test_collector_cannot_be_polled_before_start(tmp_path):
    collector = StatsCollector(CounterConnector(), [("s1", "s3")], 1, tmp_path / "samples.csv")
    with pytest.raises(RuntimeError, match="Start the stats collector"):
        collector.poll_once()
