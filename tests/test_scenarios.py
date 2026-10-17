import pytest

from congestion_project.scenarios import BurstLoad, PeriodicLoad, RampLoad


def test_ramp_profile_increases_holds_and_decreases():
    scenario = RampLoad(100, 20, ramp_in_sec=20, hold_sec=40, ramp_out_sec=40)
    assert scenario.get_bandwidth_at_time(0) == 0
    assert scenario.get_bandwidth_at_time(10) == 10
    assert scenario.get_bandwidth_at_time(30) == 20
    assert scenario.get_bandwidth_at_time(80) == 10
    assert scenario.get_bandwidth_at_time(100) == 0


def test_burst_profile_is_off_then_on():
    scenario = BurstLoad(40, 15, burst_in_sec=5, hold_sec=20, burst_out_sec=10)
    assert scenario.get_bandwidth_at_time(0) == 0
    assert scenario.get_bandwidth_at_time(5) == 15
    assert scenario.get_bandwidth_at_time(30) == 7.5
    assert scenario.get_bandwidth_at_time(35) == 0


def test_periodic_profile_validates_parameters():
    scenario = PeriodicLoad(60, 20, period_sec=20, min_bw_mbps=5, max_bw_mbps=20)
    assert scenario.get_bandwidth_at_time(0) == pytest.approx(12.5)
    assert scenario.get_bandwidth_at_time(5) == pytest.approx(20)
    with pytest.raises(ValueError):
        PeriodicLoad(60, 20, period_sec=0).get_bandwidth_at_time(1)


def test_scenario_rejects_invalid_duration():
    with pytest.raises(ValueError):
        RampLoad(0, 10)
