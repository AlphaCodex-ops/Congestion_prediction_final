# Deployment and real-network connector

## Local Windows development

The synthetic demo, feature pipeline, model training, decision engine, and unit
tests run on Windows. They do not need WSL, Mininet, Docker, or elevated
privileges.

## Before using Mininet

The PRD's host recommendation is Ubuntu (native, a Linux VM, or WSL2 where the
required virtual networking works). Install Mininet/Open vSwitch and an
iperf-compatible traffic generator only after the Linux environment has been
selected. Validate Mininet networking independently before integrating it.
WSL2 support for OVS, namespaces, and virtual networking depends on the Windows,
WSL kernel, and service setup; a dedicated Ubuntu VM is a fallback if those
features are unavailable.

## Choose a routing plane

The original requested topology mixed OVS switches and OSPF costs. OVS alone
is not an OSPF router. This guide follows Option A, FRRouting in Linux router
namespaces; the project connector does not implement Option B.

### Option A: OSPF/FRRouting

Run FRRouting on Linux routers (or router namespaces). The project provides
`MininetOspfConnector` for an already-running Mininet topology. The caller must
pass its router objects, each router's private FRR VTY socket, host IPv4
addresses and attached routers, both endpoint interface names for every link,
link capacities, and the topology's current OSPF costs. The connector does not
create Mininet, start or stop FRR, or connect to production routers.

For the three-router lab described in the guided setup, the mapping is:

```python
connector = MininetOspfConnector(
    routers={"r1": r1, "r2": r2, "r3": r3},
    link_interfaces={
        ("r1", "r3"): {"r1": "r1-eth2", "r3": "r3-eth1"},
        ("r1", "r2"): {"r1": "r1-eth1", "r2": "r2-eth0"},
        ("r2", "r3"): {"r2": "r2-eth1", "r3": "r3-eth0"},
    },
    vty_sockets={
        "r1": "/tmp/frr-mininet-lab/r1/vty",
        "r2": "/tmp/frr-mininet-lab/r2/vty",
        "r3": "/tmp/frr-mininet-lab/r3/vty",
    },
    host_endpoints={
        "h1": ("r1", "192.168.1.2"),
        "h2": ("r3", "192.168.3.2"),
    },
    link_capacities_mbps={
        ("r1", "r3"): 10,
        ("r1", "r2"): 10,
        ("r2", "r3"): 10,
    },
    initial_costs={
        ("r1", "r3"): 100,
        ("r1", "r2"): 10,
        ("r2", "r3"): 10,
    },
)
```

Import `MininetOspfConnector` from `congestion_project.connectors` in the
Python process that owns the Mininet topology; Mininet's router nodes provide
the namespace-aware `pexec` operation the adapter uses. A project `Actuator`
can then use logical link names such as `r1-r3`. The configured initial costs
must match FRR's current running costs, since they are used to undo partial
updates and by `Actuator.rollback()`.

The adapter reads `ip -j -s link` counters from the source router's interface.
It computes utilization from transmit-byte deltas and the configured link
capacity (the first sample is zero; queue depth is not collected). For OSPF
cost changes, it updates and verifies both endpoint interfaces. Updates are
runtime-only and disappear when the lab stops; they are not written to FRR
configuration files. The adapter's `close()` only disables that adapter
instance; the owner must stop the Mininet topology and its FRR processes.

The `10` Mbps capacities above are example values, not measurements. The
walkthrough topology uses ordinary Mininet links without bandwidth shaping, so
its calculated utilization is only illustrative until link capacities are
measured or explicitly shaped and the mapping is updated.

Constructing this connector alone does not start the prediction controller.
First validate connector counters, route tracing, and reversible cost changes
in the isolated lab. Any controller use must remain limited to this disposable
Mininet topology.

### Optional controller loop in the isolated Mininet lab

The caller-owned guided lab script can run prediction in `dry-run` mode or
explicitly enable lab-only route changes. The controller requires the configured
primary and backup routes and host endpoints, observes both the primary and
backup links, verifies route paths after cost changes, and restores the
original costs when the lab exits. A failed route check stops the loop and
restores the previous costs. The live-loop telemetry and decisions are written
to separate files so the training captures are not overwritten.

Use the project virtual environment for the model dependencies while adding
Ubuntu's system `dist-packages` directory to `PYTHONPATH` for Mininet:

```bash
cd /home/johan/work/Congestion_prediction
sudo env PYTHONPATH=/usr/lib/python3/dist-packages:/home/johan/work/Congestion_prediction \
  /home/johan/work/Congestion_prediction/.venv/bin/python3 \
  /home/johan/frr-mininet-lab/ospf_lab.py \
  --controller-mode dry-run --shape-primary
```

The lab script skips its long training-data capture unless `--capture` is
provided. Controller modes start a local browser dashboard in the same process.
After the script prints its dashboard URL, open `http://127.0.0.1:8765` in a
Windows or Ubuntu browser. The page displays measured utilization for both
candidate links, the model's current congestion probability, the route read
from the lab, and the latest controller decision. The lab includes h3 on a
separate LAN attached to r2. A slider controls h3's offered UDP load to h2
(0–2 Mbps); this external flow shares the shaped r2–r3 link with a managed
0.4 Mbps h1-to-h2 service flow. The controller measures r2–r3 and can change
its OSPF cost so both endpoint routes can use the r1–r3 backup. Set the slider
back to 0 to stop external traffic and demonstrate recovery.

The dashboard binds only to loopback and is available only while the lab
process is running. In the default dry-run command, the slider exercises
telemetry and decisions without changing OSPF. The solid h1 path and dashed h3
path are route lookups from Mininet, not simulated animations. In actuation
mode, only use the slider in this isolated lab and observe both route lookups
and decision verification before presenting a reroute. This external load is
generated inside the lab; it is not internet traffic.

The current Mininet controller uses a 6-second minimum reroute hold and a
6-second route-change cooldown so a classroom demonstration can show route
changes quickly. These short intervals increase the risk of switching back
and forth when the measured link becomes quiet after rerouting. Keep actuation
inside the disposable lab and monitor both the route and controller decision.

The same test flow can also be started manually at the Mininet prompt and its
route inspected with:

```text
h1 iperf3 -c 192.168.3.2 -u -b 1.7M -t 240 > /tmp/frr-mininet-lab/auto-traffic.log 2>&1 &
r1 ip route get 192.168.3.2
```

Dry-run decisions are logged to `data/results/mininet_live_decisions.csv`; no
OSPF costs are changed. Review the log and route behavior first.

Only after reviewing the dry-run, route maps, original costs, and cleanup
behavior, explicitly arm a disposable Mininet run:

```bash
sudo env PYTHONPATH=/usr/lib/python3/dist-packages:/home/johan/work/Congestion_prediction \
  /home/johan/work/Congestion_prediction/.venv/bin/python3 \
  /home/johan/frr-mininet-lab/ospf_lab.py \
  --controller-mode actuate --shape-primary --confirm-lab-only
```

The actuation flags are required together. This mode is restricted to the
Mininet nodes and OSPF links supplied by the script and must not be pointed at
production routers. At exit, confirm the script reports controller shutdown,
baseline route restoration, and shaper cleanup (`noqueue`).

### Option B: OpenFlow/SDN

Keep OVS switches and choose an SDN controller. Implement route selection by
installing/verifying flows via the controller API. Rename/control route policy
rather than claiming that an OSPF cost changed.

## Integration acceptance checks

1. Boot a disposable topology and ping hosts through the initial route.
2. Read interface counters twice and verify nonnegative deltas/utilization.
3. Change one permitted route cost and prove the actual route changed.
4. Restore the original route and verify it returned.
5. Inject connector errors; verify controller status exposes errors and costs
   roll back.
6. Run `pytest`; then perform repeatable traffic experiments and compare
   independently collected static, reactive, and predictive CSVs.

Do not run controller commands against a real network until an operator has
reviewed access controls, route limits, change authorization, and rollback.
