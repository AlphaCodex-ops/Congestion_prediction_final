import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from congestion_project.dashboard import LiveDashboard


def test_dashboard_serves_live_state_and_traffic_controls():
    actions = []
    dashboard = LiveDashboard(
        lambda: {
            "mode": "DRY-RUN",
            "actual_path": ["r1", "r2", "r3"],
            "links": {"r1-r2": [{"utilization_pct": 82.0}]},
        },
        lambda action, rate: (actions.append((action, rate)), action)[1],
        port=0,
    )
    dashboard.start()
    try:
        with urlopen(dashboard.url) as response:
            page = response.read().decode()
        assert "Live topology and selected path" in page
        assert "Offered external load from h3" in page

        with urlopen(f"{dashboard.url}/api/state") as response:
            state = json.load(response)
        assert state["actual_path"] == ["r1", "r2", "r3"]

        request = Request(
            f"{dashboard.url}/api/traffic",
            data=json.dumps({"action": "start"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert json.load(response) == {"message": "start"}
        assert actions == [("start", None)]

        request = Request(
            f"{dashboard.url}/api/traffic",
            data=json.dumps(
                {"action": "set_external_rate", "rate_mbps": 1.4}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert json.load(response) == {"message": "set_external_rate"}
        assert actions[-1] == ("set_external_rate", 1.4)

        request = Request(
            f"{dashboard.url}/api/traffic",
            data=json.dumps(
                {"action": "set_external_rate", "rate_mbps": 2.1}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
    finally:
        dashboard.stop()


def test_dashboard_rejects_non_loopback_binding():
    with pytest.raises(ValueError, match="loopback"):
        LiveDashboard(lambda: {}, lambda action, rate: action, host="0.0.0.0")


def test_dashboard_reports_snapshot_failures():
    def fail():
        raise RuntimeError("connector unavailable")

    dashboard = LiveDashboard(fail, lambda action, rate: action, port=0)
    dashboard.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"{dashboard.url}/api/state")
        assert error.value.code == 500
        assert json.load(error.value)["error"] == "connector unavailable"
    finally:
        dashboard.stop()
