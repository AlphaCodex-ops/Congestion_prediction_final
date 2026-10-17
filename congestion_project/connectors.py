from __future__ import annotations

import ipaddress
import json
import math
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LinkCounters:
    timestamp: float
    src_switch: str
    dst_switch: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors: int = 0
    dropped: int = 0
    utilization_pct: float = 0.0
    queue_depth: int = 0


class NetworkConnector(Protocol):
    """Boundary between prediction/control code and a concrete network."""

    def read_link_counters(self, src_switch: str, dst_switch: str) -> LinkCounters: ...

    def set_link_cost(self, src_switch: str, dst_switch: str, cost: int) -> None: ...

    def get_route(self, src_host: str, dst_host: str) -> list[str]: ...

    def close(self) -> None: ...


class MininetNode(Protocol):
    """Small subset of the Mininet node API used by the FRR connector."""

    def pexec(self, *args: str) -> tuple[str | bytes, str | bytes, int]: ...


class MininetOspfConnector:
    """Control FRRouting daemons running inside an owned Mininet topology.

    The topology owner supplies Mininet router nodes and explicit mappings.
    FRR commands are sent through each router's private vty socket; link
    counters and route lookups run in that router's network namespace.

    This adapter does not start or stop Mininet or FRR processes. Its ``close``
    method only prevents further operations; topology lifecycle remains with
    the caller.
    """

    def __init__(
        self,
        *,
        routers: Mapping[str, MininetNode],
        link_interfaces: Mapping[tuple[str, str], Mapping[str, str]],
        vty_sockets: Mapping[str, str | Path],
        host_endpoints: Mapping[str, tuple[str, str]],
        link_capacities_mbps: Mapping[tuple[str, str], float],
        initial_costs: Mapping[tuple[str, str], int],
    ) -> None:
        if not routers:
            raise ValueError("At least one Mininet router is required.")
        self.routers = dict(routers)
        self.vty_sockets = {name: str(path) for name, path in vty_sockets.items()}
        self.host_endpoints = dict(host_endpoints)
        self._links: dict[frozenset[str], dict[str, str]] = {}
        self._capacities: dict[frozenset[str], float] = {}
        self._costs: dict[frozenset[str], int] = {}
        self._interface_neighbors: dict[tuple[str, str], str] = {}
        self._previous_counters: dict[tuple[str, str], tuple[float, int]] = {}
        self._lock = threading.RLock()
        self._closed = False

        if set(self.vty_sockets) != set(self.routers):
            raise ValueError("A private FRR VTY socket path is required for every router.")
        if not all(self.vty_sockets.values()):
            raise ValueError("FRR VTY socket paths cannot be empty.")
        for host, (router, address) in self.host_endpoints.items():
            if router not in self.routers:
                raise ValueError(f"Host {host!r} is mapped to unknown router {router!r}.")
            try:
                ipaddress.IPv4Address(address)
            except ipaddress.AddressValueError as exc:
                raise ValueError(f"Host {host!r} must have a valid IPv4 address.") from exc

        if set(link_interfaces) != set(link_capacities_mbps) or set(link_interfaces) != set(
            initial_costs
        ):
            raise ValueError(
                "Each configured link must have interfaces, a capacity, and an initial cost."
            )
        for link, interfaces in link_interfaces.items():
            if len(link) != 2 or link[0] == link[1]:
                raise ValueError(f"Invalid router link {link!r}; expected two distinct routers.")
            endpoints = frozenset(link)
            if len(endpoints) != 2 or endpoints in self._links:
                raise ValueError(f"Router link {link!r} is duplicated or invalid.")
            if endpoints.difference(self.routers):
                raise ValueError(f"Router link {link!r} references an unknown router.")
            if set(interfaces) != endpoints or not all(
                re.fullmatch(r"[A-Za-z0-9_.:-]+", interface)
                for interface in interfaces.values()
            ):
                raise ValueError(
                    f"Link {link!r} must map each endpoint router to a valid local interface."
                )
            capacity = link_capacities_mbps[link]
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, (int, float))
                or not math.isfinite(capacity)
                or capacity <= 0
            ):
                raise ValueError(f"Link {link!r} capacity must be a positive finite Mbps value.")
            cost = initial_costs[link]
            self._validate_cost(cost)
            local_interfaces = dict(interfaces)
            self._links[endpoints] = local_interfaces
            self._capacities[endpoints] = float(capacity)
            self._costs[endpoints] = cost
            first, second = link
            self._interface_neighbors[(first, local_interfaces[first])] = second
            self._interface_neighbors[(second, local_interfaces[second])] = first

    @staticmethod
    def _validate_cost(cost: int) -> None:
        if isinstance(cost, bool) or not isinstance(cost, int) or not 1 <= cost <= 65535:
            raise ValueError(f"OSPF cost must be an integer between 1 and 65535: {cost!r}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Mininet OSPF connector is closed.")

    def _run(self, router: str, *command: str) -> str:
        self._ensure_open()
        try:
            stdout, stderr, returncode = self.routers[router].pexec(*command)
        except Exception as exc:
            raise RuntimeError(
                f"Could not run {command[0]!r} in Mininet router {router!r}."
            ) from exc
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if returncode != 0:
            detail = str(stderr).strip() or str(stdout).strip() or "no command output"
            raise RuntimeError(
                f"Command {command[0]!r} failed in router {router!r}: {detail}"
            )
        return str(stdout)

    @staticmethod
    def _link_key(src_router: str, dst_router: str) -> frozenset[str]:
        key = frozenset((src_router, dst_router))
        if len(key) != 2:
            raise ValueError("A link must connect two distinct routers.")
        return key

    def _vty_command(self, router: str, *commands: str) -> str:
        args = ["vtysh", "--vty_socket", self.vty_sockets[router]]
        for command in commands:
            args.extend(("-c", command))
        return self._run(router, *args)

    def _apply_endpoint_cost(self, router: str, interface: str, cost: int) -> None:
        self._vty_command(
            router,
            "configure terminal",
            f"interface {interface}",
            f"ip ospf cost {cost}",
        )
        output = self._vty_command(router, f"show ip ospf interface {interface}")
        match = re.search(r"\bCost:\s*(\d+)\b", output)
        if match is None or int(match.group(1)) != cost:
            raise RuntimeError(
                f"FRR did not confirm OSPF cost {cost} on {router}:{interface}."
            )

    def read_link_counters(self, src_switch: str, dst_switch: str) -> LinkCounters:
        key = self._link_key(src_switch, dst_switch)
        try:
            interfaces = self._links[key]
            capacity = self._capacities[key]
        except KeyError as exc:
            raise ValueError(f"Link {src_switch!r}-{dst_switch!r} is not configured.") from exc
        interface = interfaces.get(src_switch)
        if interface is None:
            raise ValueError(f"Link {src_switch!r}-{dst_switch!r} has no source interface.")
        output = self._run(
            src_switch, "ip", "-json", "-s", "link", "show", "dev", interface
        )
        try:
            links = json.loads(output)
            stats = links[0].get("stats64", links[0].get("stats"))
            tx = stats["tx"]
            rx = stats["rx"]
            bytes_sent = int(tx["bytes"])
            bytes_recv = int(rx["bytes"])
            packets_sent = int(tx["packets"])
            packets_recv = int(rx["packets"])
            errors = int(tx["errors"]) + int(rx["errors"])
            dropped = int(tx["dropped"]) + int(rx["dropped"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not parse interface counters for {src_switch}:{interface}."
            ) from exc

        now = time.time()
        counter_key = (src_switch, interface)
        utilization = 0.0
        with self._lock:
            previous = self._previous_counters.get(counter_key)
            if previous is not None:
                previous_time, previous_bytes = previous
                elapsed = now - previous_time
                if elapsed > 0:
                    delta_bytes = max(0, bytes_sent - previous_bytes)
                    utilization = min(
                        100.0,
                        delta_bytes * 8 / elapsed / (capacity * 1_000_000) * 100,
                    )
            self._previous_counters[counter_key] = (now, bytes_sent)
        return LinkCounters(
            timestamp=now,
            src_switch=src_switch,
            dst_switch=dst_switch,
            bytes_sent=bytes_sent,
            bytes_recv=bytes_recv,
            packets_sent=packets_sent,
            packets_recv=packets_recv,
            errors=errors,
            dropped=dropped,
            utilization_pct=utilization,
        )

    def set_link_cost(self, src_switch: str, dst_switch: str, cost: int) -> None:
        self._validate_cost(cost)
        key = self._link_key(src_switch, dst_switch)
        with self._lock:
            self._ensure_open()
            try:
                interfaces = self._links[key]
                previous_cost = self._costs[key]
            except KeyError as exc:
                raise ValueError(
                    f"Link {src_switch!r}-{dst_switch!r} is not in the connector allowlist."
                ) from exc

            attempted: list[tuple[str, str]] = []
            try:
                for router in (src_switch, dst_switch):
                    attempted.append((router, interfaces[router]))
                    self._apply_endpoint_cost(router, interfaces[router], cost)
            except Exception as exc:
                rollback_errors: list[Exception] = []
                for router, interface in reversed(attempted):
                    try:
                        self._apply_endpoint_cost(router, interface, previous_cost)
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise RuntimeError(
                        f"Setting OSPF cost failed and {len(rollback_errors)} endpoint "
                        "rollback operation(s) also failed."
                    ) from exc
                raise
            self._costs[key] = cost

    def get_route(self, src_host: str, dst_host: str) -> list[str]:
        try:
            source_router, _ = self.host_endpoints[src_host]
            destination_router, destination_address = self.host_endpoints[dst_host]
        except KeyError as exc:
            raise ValueError(f"Host {exc.args[0]!r} is not configured.") from exc
        route = [source_router]
        current_router = source_router
        for _ in range(len(self.routers)):
            if current_router == destination_router:
                return route
            output = self._run(
                current_router,
                "ip",
                "-json",
                "route",
                "get",
                destination_address,
            )
            try:
                route_info = json.loads(output)[0]
                interface = str(route_info["dev"])
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Could not determine route from router {current_router!r} "
                    f"to host {dst_host!r}."
                ) from exc
            next_router = self._interface_neighbors.get((current_router, interface))
            if next_router is None:
                raise RuntimeError(
                    f"Router {current_router!r} routes toward {dst_host!r} over "
                    f"unmapped interface {interface!r}."
                )
            if next_router in route:
                raise RuntimeError(f"Routing loop detected while tracing route {route!r}.")
            route.append(next_router)
            current_router = next_router
        raise RuntimeError(
            f"Could not trace a route from {src_host!r} to {dst_host!r} "
            f"within {len(self.routers)} routers."
        )

    def close(self) -> None:
        """Disable this adapter without stopping caller-owned Mininet/FRR processes."""
        with self._lock:
            self._closed = True
