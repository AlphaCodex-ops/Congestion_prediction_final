# Predictive Congestion Detection and OSPF Route Control

## Abstract

This project is a network research prototype that combines measured link
telemetry, short-horizon congestion prediction, an OSPF route-change policy,
and a live dashboard. Its network testbed is a Linux Mininet topology running
FRRouting (FRR). The learning pipeline is written in Python and uses
scikit-learn to predict whether a link's utilization will exceed a configured
threshold during the next ten seconds. If the prediction or a reactive safety
rule calls for a change, the controller can raise the OSPF cost of the
congested path and prefer a configured alternate path.

The project demonstrates an end-to-end design in an isolated lab; it is not a
production routing system. Historical training and validation captures measure
the `r1-r2` link. The current live demonstration instead monitors and controls
`r2-r3`. The saved model currently loaded by the lab was trained on `r1-r2`,
so its live probabilities on `r2-r3` are not yet validated. This distinction
is essential when interpreting the impressive offline scores or presenting
the dashboard.

## 1. Project purpose and scope

### 1.1 Problem statement

Congestion can build before a link reaches its capacity. A purely reactive
system waits until a utilization or loss limit has already been crossed. This
prototype explores whether recent telemetry can predict a near-future
utilization threshold crossing and use that warning to select an alternate
route.

The intended pipeline is:

1. Generate controlled traffic in a small virtual network.
2. Measure traffic counters on selected router interfaces.
3. Turn the time-ordered telemetry into model features and future labels.
4. Train and evaluate a classifier on captured CSV data.
5. Calculate a congestion probability on live samples.
6. Apply controller safeguards and, only when explicitly armed, update FRR's
   OSPF interface costs.
7. Display the measurements, prediction, decision, and actual route in a local
   browser dashboard.

### 1.2 What the project demonstrates

- Mininet hosts and routers communicating through Linux virtual interfaces.
- FRR establishing OSPF adjacencies and selecting routes from link costs.
- Linux traffic shaping to create a repeatable narrow link.
- Telemetry collection, feature generation, scenario-aware data splitting,
  model training, and offline validation.
- Guarded dry-run and lab-only route actuation with route verification and
  rollback.
- A local dashboard that shows observed network state and can request bounded
  external offered load from a host inside the lab.

### 1.3 What the results do not establish

- They do not establish performance on physical routers or a production
  network.
- They do not show that Mininet reproduces every hardware, queueing, timing,
  or control-plane behavior of a real network.
- The simulated policy-comparison output is not a packet-level or measured
  Mininet benchmark.
- The existing saved-model scores do not validate prediction on the current
  `r2-r3` link or the dashboard-controlled `h3` experiment.
- A high offline score does not by itself prove that predictive routing
  improves end-user throughput, latency, or packet loss.

## 2. Technology primer

| Technology | What it is | How it is used here |
|---|---|---|
| **Mininet** | A network emulator that creates virtual hosts, switches or routers, and links using Linux processes, network namespaces, and virtual Ethernet pairs. | Runs the topology on one Linux host while giving each network node an isolated network view. |
| **Linux network namespace** | A Linux isolation mechanism for network interfaces, routing tables, and processes. | Each Mininet router/host has its own interfaces and routes. The connector executes commands inside the appropriate router namespace. |
| **veth pair** | A pair of virtual Ethernet interfaces; packets sent into one end appear at the other. | Connects Mininet nodes and carries the lab's packets. |
| **FRRouting (FRR)** | An open-source routing software suite. `zebra` manages interface and route information; `ospfd` implements OSPF. | Runs routing processes in the three router namespaces. |
| **OSPF** | A link-state interior gateway protocol. Routers exchange topology information and compute shortest paths using configured link costs. | The project changes interface costs so the route computation prefers the alternate path. |
| **Open vSwitch (OVS)** | A virtual software switch, often controlled by OpenFlow. It does not implement OSPF routing by itself. | A basic Mininet ping test may use an OVS bridge, but the routed experiment uses Linux routers and FRR, not OSPF on OVS. |
| **`tc` / TBF** | Linux traffic-control utilities; Token Bucket Filter can cap a network interface's transmit rate. | Shapes `r2-eth1`, the `r2-r3` link endpoint, to 2 Mbps to create the experimental bottleneck. |
| **iperf3** | A traffic-generation and throughput measurement tool. | Sends controlled UDP streams from `h1` and `h3` toward `h2`. |
| **Python, pandas, NumPy** | Programming and data-analysis tools. | Read telemetry, calculate features and labels, train/evaluate the model, and provide the controller and HTTP dashboard. |
| **scikit-learn** | A machine-learning library. | Supplies `GradientBoostingClassifier`, `StandardScaler`, and classification metrics. |
| **joblib** | Python object persistence commonly used for scikit-learn artifacts. | Saves the trained model bundle and scaler so the same artifacts can be loaded for inference. |
| **Local HTTP dashboard** | A browser interface served by the lab process. | Polls local JSON state and sends traffic-rate actions. It binds to loopback (`127.0.0.1`) rather than exposing a public service. |

Mininet is an emulator, not Cisco Packet Tracer and not a connection to a
physical Cisco network. It runs real Linux networking tools and software in
virtualized nodes, making it useful for controlled experiments and
demonstrations. The topology and traffic still share the host machine's
resources.

### 2.1 How packets and routes work in this lab

Mininet starts each virtual host/router as a process in a Linux network
namespace. A virtual Ethernet pair connects the namespace interfaces that
represent each link. When `h1` sends an IP packet to `h2`, the packet leaves
`h1` through its default gateway, is routed by `r1`, traverses the router
interfaces selected by the routing table, and reaches the destination subnet
behind `r3`.

FRR's `ospfd` sends OSPF Hello packets to discover neighboring routers and
maintain adjacencies. Routers exchange link-state advertisements describing
their connected links and costs. Each router builds a link-state database and
runs a shortest-path calculation over it; the lowest-cost route to the
destination network is installed in Linux's routing table. Raising the
`r2-r3` cost changes the path calculation. It is a control-plane operation:
FRR recomputes routes and Linux forwards subsequent packets using the new
route. It does not rewrite or reroute each packet individually in the Python
controller.

The `tc` TBF qdisc rate-limits packets leaving `r2-eth1`. If traffic offers
more than 2 Mbps, packets can wait in the qdisc or be dropped according to
its queue configuration; the actual delivered throughput is not simply the
iperf3 requested rate. The collector observes Linux interface counters over
time, converts byte deltas into a rate, and normalizes that rate by the
configured link capacity to produce utilization. This is why the link and
capacity mappings must match the physical virtual interface being measured.

`iperf3` is the traffic source, not the congestion detector. Its UDP
bandwidth option specifies an offered load; the shaped link, other concurrent
traffic, and host scheduling determine what actually crosses the interface.
The model consumes measured counters, not the slider value itself.

## 3. Experimental topology

The current guided lab contains three Linux routers, three hosts, and two
possible routes between `r1` and `r3`:

```text
                    10 Mbps
               +---------------+
               |               |
h1 -- r1 ------ r2 ======= r3 -- h2
      \                         /
       \------- direct --------/
               10 Mbps

h3 -- r2
```

The drawing is conceptual; exact interfaces, addresses, paths, and costs are
listed below. `r2-r3` is the shaped 2 Mbps bottleneck (shown with `=====`).
`r1-r3` is the direct, 10 Mbps alternate router link.

### 3.1 Addressing and route design

| Segment | Addressing / interfaces | Capacity in connector | Initial OSPF cost |
|---|---|---:|---:|
| `h1-r1` LAN | `h1`: `192.168.1.2/24`; `r1`: `192.168.1.1/24` | Not a routed backbone link | — |
| `r1-r2` | `10.0.12.0/30`; `r1-eth1` / `r2-eth0` | 10 Mbps mapping | 10 |
| `r2-r3` | `10.0.23.0/30`; `r2-eth1` / `r3-eth0` | 2 Mbps; shaped at `r2-eth1` | 10 |
| `r1-r3` | `10.0.13.0/30`; `r1-eth2` / `r3-eth1` | 10 Mbps mapping | 100 |
| `r3-h2` LAN | `r3`: `192.168.3.1/24`; `h2`: `192.168.3.2/24` | Not a routed backbone link | — |
| `h3-r2` LAN | `r2`: `192.168.2.1/24`; `h3`: `192.168.2.2/24` | Not a routed backbone link | — |

The intended initial `h1`-to-`h2` route is:

```text
h1 -> r1 -> r2 -> r3 -> h2
```

The alternate is:

```text
h1 -> r1 -> r3 -> h2
```

The initial backbone cost through `r2` is lower than the direct `r1-r3` cost,
so OSPF prefers the primary route. To request a reroute, the policy raises the
`r2-r3` cost to 999 and lowers the `r1-r3` cost to 10. OSPF then prefers the
direct `r1-r3` path for `h1` to `h2`. The host `h3` is attached to `r2` and
sends a second flow toward `h2`; when `r2-r3` is made unattractive, its route
can also traverse `r1-r3`.

OSPF costs are administrative path-selection weights, not Mbps values. A
cost of 999 does not mean a 999 Mbps link or physically disable a link. The
controller changes the runtime OSPF interface costs at both ends of the
configured router link and verifies the resulting route. The lab restores its
original costs during shutdown.

### 3.2 Traffic sources and shaping

- `h1` sends a managed UDP service flow to `h2` at approximately 0.4 Mbps.
- `h3` sends a separate UDP flow to `h2`; the dashboard slider requests 0 to
  2 Mbps in bounded increments.
- The two flows share the shaped `r2-r3` link while their routes use it.
- The TBF qdisc caps `r2-eth1` at 2 Mbps. The lab attempts cleanup and checks
  that the qdisc is removed on exit.
- The dashboard's `h3` flow is generated inside Mininet. It is not incoming
  Internet traffic.

These parameters make congestion observable and controllable. The 2 Mbps
bottleneck is deliberately modest so a small, repeatable UDP load can push it
toward capacity on a laptop.

## 4. Telemetry and data preparation

### 4.1 What is measured

The Mininet/FRR connector reads interface counters in the router namespace.
The collector records timestamps, router/interface identity, byte and packet
counters, drop/error counts, derived byte and packet rates, utilization, and a
drop-rate estimate into CSV. It polls once per second in the current
configuration.

Utilization is derived from the measured transmitted-byte rate and the
configured capacity of the mapped link. Therefore, both the counter endpoint
and configured link capacity must match the interface being measured. A wrong
capacity or wrong link mapping changes the utilization percentage and can
invalidate labels and predictions.

The first sample for a link has no prior counter reading from which to compute
a rate, so its rate-derived fields start at zero. Counter wrap handling and
time deltas are accounted for by the collector/connector. The connector's
`queue_depth` is not a direct Linux qdisc backlog measurement in this
integration; it should not be interpreted as a measured queue length.

### 4.2 Capture inventory

| File | Raw rows | Distinct scenarios | Role and measured link |
|---|---:|---:|---|
| `data/raw/mininet_training_run1.csv` | 731 | 12 | First labeled Mininet capture; `r1-r2`, `r1-eth1`. |
| `data/raw/mininet_training_run2.csv` | 1,096 | 18 | Added near-threshold scenarios; `r1-r2`, `r1-eth1`. |
| `data/raw/mininet_training_run3.csv` | 1,460 | 24 | Added below/near/above-threshold scenarios; `r1-r2`, `r1-eth1`. |
| `data/raw/mininet_development_run1-3.csv` | 3,287 | 54 | Combined development CSV; scenario names are prefixed by capture (`run1::`, etc.) to keep capture scenarios separate. |
| `data/raw/mininet_training_run4.csv` | 1,462 | 24 | Separate later capture used for model-development validation; `r1-r2`, `r1-eth1`. |
| `data/raw/mininet_live_controller.csv` | 3,518 | No scenario column | Live-controller telemetry; contains rows for `r1-r2`, `r1-r3`, and `r2-r3` from different observation intervals. It is not a scenario-labeled training/validation set. |
| `data/raw/synthetic_training.csv` | 360 | 3 | Deterministic synthetic `ramp`, `burst`, and `periodic` profiles; software-pipeline testing only. |

Each labeled Mininet CSV row is a telemetry sample, not an independent
experiment. Multiple rows within a scenario are temporally correlated.

### 4.3 Captured utilization profiles

The raw capture summaries show that the first three captures contained
increasingly fine-grained offered-load cases:

| Capture | Scenario type | Approximate mean measured utilization |
|---|---|---:|
| Run 1 | Clear | 50.6% |
| Run 1 | Congested | 98.9% |
| Run 2 | Clear | 50.6% |
| Run 2 | Near threshold | 81.0% |
| Run 2 | Congested | 98.9% |
| Run 3 | Below threshold | 65.8–66.9% |
| Run 3 | Near threshold low | 70.9–72.0% |
| Run 3 | Near threshold high | 75.9% |
| Run 3 | Above threshold | 86.0–87.5% |
| Run 4 | Below threshold | 68.3% |
| Run 4 | Near threshold low | 73.4% |
| Run 4 | Near threshold high | 78.4–78.5% |
| Run 4 | Above threshold | 83.5–84.9% |

Some raw scenario minimums are close to zero even when their means are high.
This reflects the short starting/transition region in the collected samples;
feature generation excludes the initial history and future-label warm-up
region. The table reports means over raw rows, not the training feature rows.

## 5. Feature engineering and target

### 5.1 Prediction question

The saved Mininet models use a 1-second sample interval, utilization threshold
of 70%, and forecast horizon of 10 seconds. For a sample at time `t`, the
binary target is:

```text
1 (congested) if the maximum measured utilization in the next 10 seconds is > 70%
0 (clear) otherwise
```

The threshold is strictly greater than 70%, not greater-than-or-equal. This
target asks whether the next short window contains a threshold crossing; it
does not predict a numeric future utilization value.

Rows are grouped by scenario before lagging and labeling, so a rolling window
or forecast does not continue from one scenario into another. The scenario
identifier is also used to hold out whole scenarios in the final grouped model.

### 5.2 The 15 features in the saved Mininet model

| Feature | Meaning |
|---|---|
| `utilization_pct` | Current measured utilization at time `t`. |
| `utilization_pct_lag1` | Utilization one sample (one second) earlier. |
| `utilization_pct_lag2` | Utilization two samples earlier. |
| `utilization_pct_lag5` | Utilization five samples earlier. |
| `utilization_rolling_mean_5s` | Mean utilization in the trailing five samples. |
| `utilization_rolling_max_5s` | Maximum utilization in the trailing five samples. |
| `utilization_rolling_std_5s` | Standard deviation in the trailing five samples. |
| `utilization_rolling_mean_10s` | Mean utilization in the trailing ten samples. |
| `utilization_rolling_max_10s` | Maximum utilization in the trailing ten samples. |
| `utilization_rolling_std_10s` | Standard deviation in the trailing ten samples. |
| `utilization_delta` | Current utilization minus the previous sample. |
| `utilization_accel` | Change in the utilization delta relative to the preceding delta. |
| `packets_per_sec` | Current measured packet rate. |
| `packets_per_sec_lag1` | Packet rate one sample earlier. |
| `drop_rate` | Fraction of packets recorded as dropped over the sample interval. |

Using current values plus recent history gives the model information about
level, persistence, and short-term changes. Rolling means smooth brief
counter noise; rolling maxima help retain short peaks; packet rate and drop
rate add load-related context. Feature importance is a model-specific split
statistic, not proof that a feature causes congestion.

### 5.3 Train/test preparation

The Mininet trainer fits a `StandardScaler` on training features and persists
it with the model. The scaler is then reused unchanged for inference. This
avoids fitting transformations on test samples and keeps offline/online
feature values on the same scale. Scaling is not a requirement of
gradient-boosted trees, but retaining it provides a consistent pipeline and
is useful if the model family changes.

Training uses scikit-learn's `GradientBoostingClassifier` with the current
configured values:

| Parameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | 120 | Number of boosting stages. |
| `learning_rate` | 0.05 | Contribution of each stage. |
| `max_depth` | 3 | Limits the complexity of each tree. |
| `min_samples_split` | 10 | Minimum samples required to split a node. |
| `min_samples_leaf` | 5 | Minimum samples retained in a leaf. |
| `subsample` | 0.8 | Fraction of training samples used per stage. |
| `random_state` | 42 | Reproducibility for estimator randomness. |

The trainer uses balanced sample weights when configured to do so. In the
grouped development model, scenario IDs—not individual rows—are split into
training and test groups. The saved artifact contains 37 training scenario
IDs and 17 held-out scenario IDs (both subsets include scenarios from the
same three capture sessions). This is better than randomly splitting
neighboring time samples, but it is not leave-one-capture-run-out evaluation.

The fitted `GradientBoostingClassifier` builds shallow decision trees in
sequence. Each stage focuses on errors left by the preceding stages, and
their combined output is converted to a congestion probability. The
probability is not the predicted future utilization percentage; it is the
model's estimated likelihood of the binary future-threshold label. Keeping
this probability separate from the 70% utilization target allows the
controller to use policy thresholds such as 0.70 and 0.40.

## 6. Model artifacts and measured results

There are multiple saved artifacts from different stages. They must not be
treated as interchangeable. The currently referenced lab model is
`data/models/mininet_development_grouped_run1-3/model_v1.joblib` and its
matching scaler; both were trained for `r1-r2` telemetry.

### 6.1 Summary of saved model evaluations

Confusion matrices below use the order `[[true clear, false congested],
[missed congestion, true congested]]`, i.e. `[[TN, FP], [FN, TP]]`.

| Artifact | Evaluation | Samples | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `data/models/metrics.json` | Synthetic scenario holdout (`periodic`) | 108 | 0.759 | 0.759 | 1.000 | 0.863 | 0.852 | `[[0,26],[0,82]]` |
| `data/models/mininet_run1/` | Run 1 internal split | 150 test | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `[[68,0],[0,82]]` |
| `data/models/mininet_development_run1-3/` | Development split, earlier artifact | 657 test | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `[[164,0],[0,493]]` |
| `data/models/mininet_development_grouped_run1-3/` | Grouped-scenario development split; currently loaded by lab script | 694 test | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `[[164,0],[0,530]]` |

The synthetic model predicts every test row as congested: it has 26 clear
false positives and no clear true negatives. Its 75.9% accuracy equals the
majority-class baseline and should not be presented as a useful congestion
detector. It verifies that the pipeline runs on synthetic profiles; it is not
real-network evidence.

All Mininet training-set evaluation rows achieved perfect metrics, but the
test samples are still from the same testbed and capture collection as their
training data. A scenario-held-out split reduces direct time-adjacent leakage
but does not replace an independent capture from a later, untouched run or
another topology.

### 6.2 Run 4 validation for the earlier development artifact

The probability-cutoff table supplied during development matches
`data/models/mininet_development_run1-3/`, the earlier artifact. Run 4 contains
982 valid feature/label rows: 246 clear and 736 congested. Its rows come from
the `r1-r2` link. The table shows that raising the cutoff reduced false
positives without losing a congested row in this particular capture:

| Probability cutoff | TP | FN | FP | TN | Recall | Precision | False-positive rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 736 | 0 | 81 | 165 | 1.000 | 0.901 | 0.329 |
| 0.55 | 736 | 0 | 75 | 171 | 1.000 | 0.908 | 0.305 |
| 0.60 | 736 | 0 | 63 | 183 | 1.000 | 0.921 | 0.256 |
| 0.65 | 736 | 0 | 55 | 191 | 1.000 | 0.930 | 0.224 |
| 0.70 | 736 | 0 | 43 | 203 | 1.000 | 0.945 | 0.175 |
| 0.75 | 736 | 0 | 39 | 207 | 1.000 | 0.950 | 0.159 |
| 0.80 | 736 | 0 | 29 | 217 | 1.000 | 0.962 | 0.118 |
| 0.85 | 736 | 0 | 24 | 222 | 1.000 | 0.968 | 0.098 |
| 0.90 | 736 | 0 | 19 | 227 | 1.000 | 0.975 | 0.077 |
| 0.95 | 736 | 0 | 11 | 235 | 1.000 | 0.985 | 0.045 |

The false-positive rate is `FP / (FP + TN)`. Run 4 was examined to compare
cutoffs, so this table is model-development evidence. Choosing 0.95 because
it looks best on this table would tune against run 4; its apparent recall of
1.0 is not an independent guarantee for another run. Use an untouched capture
to report a final fixed-cutoff score.

### 6.3 Run 4 validation for the currently loaded grouped artifact

The grouped model used by the lab script was independently recalculated
against the same 982 run-4 rows at the historical `r1-r2` target. At the
default 0.5 cutoff:

| Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.911 | 0.894 | 1.000 | 0.944 | 1.000 | `[[159,87],[0,736]]` |
| 0.70 | 0.949 | 0.936 | 1.000 | 0.967 | 1.000 | `[[196,50],[0,736]]` |
| 0.95 | 0.993 | 0.991 | 1.000 | 0.995 | 1.000 | `[[239,7],[0,736]]` |

At 0.5, the majority-class baseline is 0.749. The model catches all 736
congested labels in this capture, but it also labels 87 of the 246 clear
samples as congested. At 0.7, false positives fall to 50 in this capture.
These numbers are not directly interchangeable with the preceding cutoff
table because they come from a different saved model artifact.

### 6.4 Feature importance for the currently loaded grouped artifact

| Feature | Importance |
|---|---:|
| `utilization_rolling_mean_5s` | 0.237699 |
| `utilization_pct` | 0.155826 |
| `packets_per_sec_lag1` | 0.139535 |
| `utilization_rolling_mean_10s` | 0.131294 |
| `utilization_pct_lag2` | 0.102889 |
| `utilization_pct_lag5` | 0.099215 |
| `utilization_pct_lag1` | 0.078636 |
| `packets_per_sec` | 0.038491 |
| `utilization_rolling_max_10s` | 0.015840 |
| `utilization_rolling_max_5s` | 0.000575 |
| `utilization_rolling_std_5s` | approximately 0 |
| `utilization_rolling_std_10s` | approximately 0 |
| `utilization_delta` | approximately 0 |
| `utilization_accel` | approximately 0 |
| `drop_rate` | 0 |

The short rolling mean, current utilization, recent packet rate, and longer
rolling mean contribute most to this fitted artifact. Near-zero importance
does not prove that a feature is universally useless; it means the fitted
trees did not use it to reduce impurity in this training split.

## 7. Online decision and safety design

### 7.1 Probability is not the same as utilization

The system uses several separate thresholds:

| Setting | Current value | Meaning |
|---|---:|---|
| Model label threshold | 70% utilization | A training label is congested if a future sample within the next 10 seconds exceeds 70%. |
| Controller high probability threshold | 0.70 | When not rerouted, a predicted congestion probability at or above 0.70 requests a reroute. |
| Controller low probability threshold | 0.40 | When rerouted, a probability at or below 0.40 can permit a return to the primary path after the hold and cooldown. |
| Reactive utilization safeguard | 85% | Requests a reroute if current utilization reaches this value, even if prediction confidence is lower. |
| Minimum reroute hold | 6 seconds | Prevents immediate restoration after switching to the alternate. |
| Route-change cooldown | 6 seconds | Prevents repeated changes closer together than the configured interval. |
| Route verification wait | 10 seconds | Allows route convergence before checking the expected path. |

The 0.70 and 0.40 values are probability thresholds, not percentages of
link utilization. Their gap is hysteresis: the controller uses a higher
threshold to move onto the backup route and a lower threshold to move back.
This reduces rapid threshold chatter. The 85% safeguard is reactive; it is
not a model probability.

### 7.2 Dry-run, actuation, and rollback

- Controller mode defaults to `off`.
- `dry-run` computes and logs candidate decisions but does not alter OSPF.
- `actuate` requires both `--shape-primary` and `--confirm-lab-only`.
- At startup, the actuator checks that the expected primary route is active.
- A requested cost change is applied through FRR and verified at both
  endpoints.
- The controller waits for convergence, checks the observed route, and rolls
  costs back if the route does not match the expected path.
- Original route costs are restored when the controller/lab is stopped.

These checks reduce accidental route changes in the disposable lab. They are
not a general authorization system for real routers; production access
controls, multi-operator safeguards, and broad failure recovery are outside
the project scope.

### 7.3 Why route swapping can recur

The current return-to-primary condition is based on the measured link's
post-reroute probability, the 6-second minimum hold, and cooldown. Once
traffic moves off `r2-r3`, that link's measured utilization may fall, even
while the dashboard slider continues sending traffic from `h3`. After both
the hold and the 6-second cooldown expire, the prediction may fall below 0.40
and permit restoration.
The restored route moves the flow back onto `r2-r3`, which can raise
utilization again and produce another reroute.

The short hold and cooldown make quicker demonstrations possible, but they
also make this feedback loop and route flapping more likely. These timers do
not prove that returning to the primary path is safe while offered load
remains high. A more robust follow-up would include the active offered-load
state and/or estimate whether the primary link can carry the flow before
restoring its cost. Such a policy should be tested against repeated traffic
traces before being claimed as fixed.

## 8. Live dashboard and available evidence

The dashboard is served by the lab process at
`http://127.0.0.1:8765` by default. It displays link telemetry, model
probability, controller decision, and route lookups from Mininet. Its external
traffic control starts/stops the bounded `h3`-to-`h2` UDP flow. The topology
visualization reflects observed routes; it is not a simulated animation.
Because the server binds to loopback, it is intended for the local machine
running the lab and is not exposed as a public dashboard.

The current `mininet_live_controller.csv` has 3,518 telemetry rows across
three links. Its measured summaries are:

| Link | Rows | Mean utilization | 95th percentile | Maximum | Samples above 70% |
|---|---:|---:|---:|---:|---:|
| `r1-r2` | 817 | 25.794% | 87.666% | 87.918% | 29.25% |
| `r1-r3` | 1,759 | 4.905% | 17.533% | 22.775% | 0% |
| `r2-r3` | 942 | 13.034% | 20.917% | 99.538% | 0.53% |

The links do not have equal row counts or identical observation windows, so
their means are not a controlled head-to-head performance comparison. The
capture has no scenario labels or traffic-slider values; the occasional high
`r2-r3` sample cannot be attributed to a particular external offered-load
setting from this file alone.

The live decision CSV contains 1,526 records: 1,423 holds, 98 reroute
decisions, and 5 revert decisions. It includes both dry-run and actuation
records. The log includes 91 reroute decisions explicitly marked dry-run,
seven observed backup-route values, and five observed primary-route values;
the verification text records seven reroute-verification-passed entries.
This is useful evidence that route checks and decision logging have been
exercised in the lab, but the accumulated log is not a single clean experiment
and does not establish measured benefit from the current external-load
scenario.

## 9. Important validation limitation: training link versus live link

This is the most important limitation for interpreting the present demo:

1. All four labeled training/validation CSVs (`run1`–`run4`) contain telemetry
   only for `r1-r2` on `r1-eth1`.
2. The currently loaded grouped model's saved metadata says
   `target_link: [r1, r2]`.
3. The current `config.mininet.yaml` sets the primary decision/monitoring link
   to `[r2, r3]`, and the online feature builder uses that primary link.
4. The current lab shapes and controls `r2-r3`.

Although the 15 feature names are the same, identical column names do not mean
the training and live distributions are the same. In particular, the
capacity, interface, traffic mix, and load patterns differ. Therefore,
displaying a probability for `r2-r3` is technically possible but the saved
model's prediction quality on that link is not validated by the run-4
results. Do not describe the current h3 slider experiment as verified
predictive congestion detection until a compatible model has been trained
and evaluated.

### Recommended validation work

1. Capture multiple labeled scenarios while measuring `r2-r3` with the same
   interface, 2 Mbps capacity, and route configuration used by the live lab.
2. Include clear, near-threshold, congested, ramp, burst, and recovery phases;
   record the h3 offered rate and scenario identifier with each sample.
3. Train the model using only development captures and confirm the saved
   `feature_config.target_link` is `["r2", "r3"]`.
4. Validate a fixed model and fixed probability threshold on a separate,
   untouched capture. Run 5 is optional for a classroom prototype, but an
   independent later capture is needed for a stronger final-performance
   claim.
5. Compare static, reactive, and predictive routing on repeatable identical
   traces, recording throughput, latency, loss, route-convergence time, and
   false reroute/return events.
6. Confirm route restoration and shaper cleanup after both normal shutdown
   and injected failure cases.

Do not solve the link mismatch by silently reusing the `r1-r2` model and
calling its `r2-r3` probabilities validated. The target-specific capture and
evaluation are the appropriate next step.

## 10. Reproducibility and operation

The detailed environment-specific startup sequence is maintained in
[`deployment.md`](deployment.md). A controller dry-run command for the
isolated Ubuntu/WSL lab is:

```bash
cd /home/johan/work/Congestion_prediction
sudo env PYTHONPATH=/usr/lib/python3/dist-packages:/home/johan/work/Congestion_prediction \
  /home/johan/work/Congestion_prediction/.venv/bin/python3 \
  /home/johan/frr-mininet-lab/ospf_lab.py \
  --controller-mode dry-run --shape-primary
```

Open the printed local dashboard URL, usually `http://127.0.0.1:8765`.
Dry-run is the recommended mode for demonstrating predictions and dashboard
controls without changing the route. To test actual route changes, use the
explicit actuation command documented in [`deployment.md`](deployment.md)
only in the disposable Mininet lab; it requires
`--controller-mode actuate --shape-primary --confirm-lab-only`.

The existing captured-data workflow and artifacts are documented in
[`training_guide.md`](training_guide.md). Use the model and scaler from the
same output directory as a matched pair. The current `config.mininet.yaml`
selects `r2-r3`, whereas historical model artifacts select `r1-r2`; do not
run a historical validation command with the current target link and assume
it evaluated the saved `r1-r2` model.

## 11. Conclusion

The project has progressed from controlled telemetry capture to an integrated
Mininet/FRR route-control demonstration with a live dashboard and bounded
traffic control. It has a working model-training pipeline, a documented
short-horizon target, and historical `r1-r2` validation results. The present
evidence is promising for a controlled classroom prototype, but the
development captures are highly structured, run 4 has already informed
threshold/model development, and the deployed model/link currently do not
match.

The defensible conclusion is that the software pipeline and guarded lab
actuation have been exercised, while model accuracy on the current `r2-r3`
external-load scenario and measurable end-user benefit remain to be
established with target-specific captures and an untouched evaluation.

## 12. Project reference files

- [`config.mininet.yaml`](../config.mininet.yaml) — current model, collector,
  topology-policy, and controller settings.
- [`data/models/mininet_development_grouped_run1-3/metrics.json`](../data/models/mininet_development_grouped_run1-3/metrics.json) — saved metadata and metrics for the model loaded by the lab script.
- [`data/models/mininet_development_run1-3/metrics.json`](../data/models/mininet_development_run1-3/metrics.json) — earlier development model associated with the supplied cutoff sweep.
- [`congestion_project/features.py`](../congestion_project/features.py) — time-series features, labels, and split strategy.
- [`congestion_project/training.py`](../congestion_project/training.py) — trainer, metrics, saved model, and predictor.
- [`congestion_project/collector.py`](../congestion_project/collector.py) — polling, rate/drop calculations, and CSV collection.
- [`congestion_project/connectors.py`](../congestion_project/connectors.py) — network connector and Mininet/FRR operations.
- [`congestion_project/control.py`](../congestion_project/control.py) — thresholds, hysteresis, actuation, verification, rollback, and logging.
- [`congestion_project/dashboard.py`](../congestion_project/dashboard.py) — loopback dashboard and traffic-control API.
- [`deployment.md`](deployment.md) — environment setup, lab commands, and safety checks.
- [`architecture.md`](architecture.md) — system component and data-flow overview.
- [`next_steps.md`](next_steps.md) — limitations and remaining experimental work.

## 13. Appendix: feature importance from every saved model artifact

The values below reproduce the saved `metrics.json` feature-importance values,
rounded to six decimal places. A dash means that the artifact did not use that
feature. These values are not directly comparable across artifacts: the
synthetic model uses a different sampling interval, feature set, target link,
and data source, while the Mininet artifacts come from different training
splits.

| Feature | Synthetic | Mininet run 1 | Development run 1–3 (earlier) | Development run 1–3 (grouped; loaded by lab) |
|---|---:|---:|---:|---:|
| `utilization_pct` | 0.903051 | 0.135186 | 0.129200 | 0.155826 |
| `utilization_pct_lag1` | 0.015273 | 0.098094 | 0.150045 | 0.078636 |
| `utilization_pct_lag2` | 0.000001 | 0.078613 | 0.083387 | 0.102889 |
| `utilization_pct_lag3` | 0.000002 | — | — | — |
| `utilization_pct_lag5` | — | 0.206440 | 0.063932 | 0.099215 |
| `utilization_pct_lag6` | 0.000004 | — | — | — |
| `utilization_rolling_mean_5s` | — | 0.005281 | 0.005238 | 0.237699 |
| `utilization_rolling_max_5s` | — | 0.082870 | 0.154729 | 0.000575 |
| `utilization_rolling_std_5s` | — | 0.000000 | 0.000000 | 0.000000 |
| `utilization_rolling_mean_10s` | — | 0.122965 | 0.005806 | 0.131294 |
| `utilization_rolling_max_10s` | — | 0.054765 | 0.060180 | 0.015840 |
| `utilization_rolling_std_10s` | — | 0.042627 | 0.000000 | 0.000000 |
| `utilization_rolling_mean_30s` | 0.000003 | — | — | — |
| `utilization_rolling_max_30s` | 0.000002 | — | — | — |
| `utilization_rolling_std_30s` | 0.000002 | — | — | — |
| `utilization_delta` | 0.000003 | 0.000000 | 0.000000 | 0.000000 |
| `utilization_accel` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `packets_per_sec` | 0.000129 | 0.122301 | 0.198158 | 0.038491 |
| `packets_per_sec_lag1` | 0.000211 | 0.050858 | 0.149326 | 0.139535 |
| `drop_rate` | 0.002438 | 0.000000 | 0.000000 | 0.000000 |

The synthetic artifact's saved feature configuration uses a 5-second interval,
lags `[1, 2, 3, 6]`, rolling windows of 10 and 30 seconds, a 30-second target
horizon, and `s1-s3` as its target link. The Mininet artifacts use the
1-second interval, lags `[1, 2, 5]`, rolling windows of 5 and 10 seconds, a
10-second horizon, and `r1-r2` as their saved target link. The currently
configured online target remains `r2-r3`, as explained in Section 9.
