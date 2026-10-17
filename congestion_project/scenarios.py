from __future__ import annotations

import math


class LoadScenario:
    """Pure load profile; a network-specific traffic generator is a separate connector."""

    def __init__(
        self,
        duration_sec: float,
        bandwidth_mbps: float,
        *,
        sample_interval_sec: float = 5.0,
        **parameters: float,
    ) -> None:
        if duration_sec <= 0 or bandwidth_mbps < 0 or sample_interval_sec <= 0:
            raise ValueError("Duration and sample interval must be positive; bandwidth cannot be negative.")
        self.duration_sec = float(duration_sec)
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.sample_interval_sec = float(sample_interval_sec)
        self.parameters = parameters

    def get_bandwidth_at_time(self, t_sec: float) -> float:
        return 0.0


class RampLoad(LoadScenario):
    def get_bandwidth_at_time(self, t_sec: float) -> float:
        ramp_in = max(0.0, float(self.parameters.get("ramp_in_sec", self.duration_sec / 3)))
        hold = max(0.0, float(self.parameters.get("hold_sec", self.duration_sec / 2)))
        ramp_out = max(0.0, float(self.parameters.get("ramp_out_sec", self.duration_sec - ramp_in - hold)))
        if t_sec < 0 or t_sec >= self.duration_sec:
            return 0.0
        if ramp_in and t_sec < ramp_in:
            return self.bandwidth_mbps * t_sec / ramp_in
        if t_sec < ramp_in + hold:
            return self.bandwidth_mbps
        if ramp_out and t_sec < ramp_in + hold + ramp_out:
            return self.bandwidth_mbps * max(0.0, 1 - (t_sec - ramp_in - hold) / ramp_out)
        return 0.0


class BurstLoad(LoadScenario):
    def get_bandwidth_at_time(self, t_sec: float) -> float:
        burst_in = max(0.0, float(self.parameters.get("burst_in_sec", 0)))
        hold = max(0.0, float(self.parameters.get("hold_sec", self.duration_sec)))
        burst_out = max(0.0, float(self.parameters.get("burst_out_sec", 0)))
        if t_sec < 0 or t_sec >= self.duration_sec or t_sec < burst_in:
            return 0.0
        if t_sec < burst_in + hold:
            return self.bandwidth_mbps
        if burst_out and t_sec < burst_in + hold + burst_out:
            return self.bandwidth_mbps * max(0.0, 1 - (t_sec - burst_in - hold) / burst_out)
        return 0.0


class PeriodicLoad(LoadScenario):
    def get_bandwidth_at_time(self, t_sec: float) -> float:
        period = float(self.parameters.get("period_sec", 20))
        minimum = float(self.parameters.get("min_bw_mbps", 0))
        maximum = float(self.parameters.get("max_bw_mbps", self.bandwidth_mbps))
        if period <= 0 or minimum < 0 or maximum < minimum:
            raise ValueError("Periodic load requires a positive period and 0 <= min bandwidth <= max bandwidth.")
        if t_sec < 0 or t_sec >= self.duration_sec:
            return 0.0
        return minimum + (maximum - minimum) * (1 + math.sin(2 * math.pi * t_sec / period)) / 2


class FailureScenario(LoadScenario):
    def get_bandwidth_at_time(self, t_sec: float) -> float:
        normal = float(self.parameters.get("normal_bw_mbps", self.bandwidth_mbps * 0.2))
        failure_at = float(self.parameters.get("failure_at_sec", self.duration_sec / 3))
        recovery_at = float(self.parameters.get("recovery_at_sec", self.duration_sec))
        if t_sec < 0 or t_sec >= self.duration_sec:
            return 0.0
        return normal if t_sec < failure_at or t_sec >= recovery_at else self.bandwidth_mbps

    @staticmethod
    def trigger_failure(connector: object, src_switch: str = "s1", dst_switch: str = "s3") -> None:
        set_cost = getattr(connector, "set_link_cost", None)
        if not callable(set_cost):
            raise TypeError("Failure simulation requires a connector with set_link_cost().")
        set_cost(src_switch, dst_switch, 999)

    @staticmethod
    def recover_failure(connector: object, src_switch: str = "s1", dst_switch: str = "s3") -> None:
        set_cost = getattr(connector, "set_link_cost", None)
        if not callable(set_cost):
            raise TypeError("Failure recovery requires a connector with set_link_cost().")
        set_cost(src_switch, dst_switch, 10)
