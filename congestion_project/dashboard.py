from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Congestion Routing Lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1220;
      --panel: #121d2e;
      --panel2: #172438;
      --line: #26364d;
      --text: #e8eef8;
      --muted: #9aabc2;
      --cyan: #4fd1e5;
      --green: #57d69a;
      --orange: #ffb454;
      --red: #ff6b70;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(ellipse at top, #14223a, var(--bg) 55%);
      color: var(--text);
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    }
    main { width: min(1200px, calc(100% - 32px)); margin: 28px auto 50px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 22px; }
    h1 { margin: 0; font-size: clamp(24px, 4vw, 36px); letter-spacing: -.04em; }
    h2 { font-size: 18px; margin: 0 0 14px; }
    p { color: var(--muted); margin: 7px 0; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; font-size: 11px; letter-spacing: .16em; font-weight: 800; }
    .badge { border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; color: var(--muted); white-space: nowrap; font-size: 12px; }
    .grid { display: grid; gap: 14px; }
    .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 14px; }
    .columns { grid-template-columns: minmax(0, 1.4fr) minmax(280px, .8fr); }
    .panel, .metric { background: linear-gradient(145deg, var(--panel2), var(--panel)); border: 1px solid var(--line); border-radius: 16px; }
    .panel { padding: 20px; margin-bottom: 14px; }
    .metric { padding: 17px 19px; min-height: 108px; }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { font-size: clamp(20px, 3vw, 28px); font-weight: 750; margin-top: 8px; letter-spacing: -.03em; overflow-wrap: anywhere; }
    .metric-detail { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .topology-wrap { overflow-x: auto; }
    .topology { width: 100%; min-width: 650px; height: auto; }
    .edge { stroke: #40516b; stroke-width: 7; stroke-linecap: round; transition: stroke .3s, opacity .3s; }
    .edge.active { stroke: var(--green); filter: drop-shadow(0 0 6px #57d69a77); }
    .edge.congested:not(.active) { stroke: var(--red); }
    .external-edge { stroke: #d5a6ff; stroke-width: 4; stroke-dasharray: 8 6; opacity: 0; }
    .external-edge.active { opacity: 1; filter: drop-shadow(0 0 5px #d5a6ff77); }
    .node { fill: #182a42; stroke: #526987; stroke-width: 2; }
    .host { fill: #173b49; stroke: #4aa9bf; }
    .node-name { fill: var(--text); font-size: 15px; font-weight: 700; text-anchor: middle; }
    .node-kind { fill: var(--muted); font-size: 10px; text-anchor: middle; }
    .edge-label { fill: #c1cee0; font-size: 12px; text-anchor: middle; paint-order: stroke; stroke: var(--panel); stroke-width: 5px; stroke-linejoin: round; }
    .edge-value { fill: var(--orange); font-weight: 700; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; color: var(--muted); font-size: 12px; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--green); }
    .dot.idle { background: #526987; }
    canvas { width: 100%; height: 260px; display: block; }
    .chart-legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; color: var(--muted); font-size: 12px; }
    .chart-line { display: inline-block; width: 18px; border-top: 3px solid var(--cyan); margin: 0 6px 3px 0; }
    .chart-line.orange { border-color: var(--orange); }
    .links { display: grid; gap: 12px; }
    .link-row { background: #0e1828; border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
    .link-head { display: flex; justify-content: space-between; gap: 10px; font-weight: 700; }
    .bar { height: 8px; border-radius: 6px; background: #27374d; overflow: hidden; margin: 9px 0 5px; }
    .bar-fill { display: block; height: 100%; background: var(--cyan); transition: width .4s, background .4s; }
    .bar-fill.high { background: var(--red); }
    .link-meta { color: var(--muted); font-size: 11px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button { cursor: pointer; border: 0; border-radius: 9px; padding: 10px 15px; color: #06101c; background: var(--green); font-weight: 800; font-size: 13px; }
    button.stop { color: var(--text); background: #34445b; }
    button:disabled { cursor: not-allowed; opacity: .45; }
    .status { color: var(--muted); font-size: 13px; }
    input[type="range"] { accent-color: var(--orange); width: min(100%, 380px); }
    .slider-row { display: grid; grid-template-columns: minmax(120px, 1fr) auto; align-items: center; gap: 12px; margin: 12px 0; }
    .rate-value { min-width: 72px; color: var(--orange); font-size: 20px; font-weight: 800; text-align: right; }
    .explain { border-left: 3px solid var(--cyan); padding: 3px 0 3px 13px; color: #d0dced; }
    .event { border-top: 1px solid var(--line); padding: 11px 0 0; margin-top: 12px; }
    .event strong { color: var(--cyan); text-transform: capitalize; }
    .error { color: var(--red); }
    .small { font-size: 12px; }
    @media (max-width: 800px) { .cards { grid-template-columns: repeat(2, 1fr); } .columns { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
    @media (max-width: 450px) { main { width: calc(100% - 20px); margin-top: 16px; } .cards { gap: 8px; } .metric { padding: 13px; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">Mininet + FRRouting · live lab</div><h1>Congestion &amp; Route Control</h1>
    <p>Measured network state, model decision, and the route currently used by h1 → h2.</p></div>
    <div class="badge" id="connection">Connecting to local lab…</div>
  </header>
  <section class="grid cards">
    <div class="metric"><div class="metric-label">CONTROLLER MODE</div><div class="metric-value" id="mode">—</div><div class="metric-detail" id="controller-state">Waiting for lab data</div></div>
    <div class="metric"><div class="metric-label">CONGESTION PROBABILITY</div><div class="metric-value" id="probability">—</div><div class="metric-detail" id="threshold">Model high threshold —</div></div>
    <div class="metric"><div class="metric-label">BOTTLENECK LINK UTILIZATION</div><div class="metric-value" id="utilization">—</div><div class="metric-detail" id="utilization-link">Measured from live interface counters</div></div>
    <div class="metric"><div class="metric-label">ACTIVE ROUTE</div><div class="metric-value" id="route">—</div><div class="metric-detail" id="route-kind">Observed from FRR / Mininet</div></div>
  </section>
  <section class="grid columns">
    <div>
      <article class="panel">
        <h2>Live topology and selected path</h2>
        <div class="topology-wrap">
          <svg class="topology" viewBox="0 0 760 360" role="img" aria-label="Mininet topology from h1 through routers r1, r2, and r3 to h2">
            <line class="edge" data-link="r1-r2" x1="270" y1="164" x2="475" y2="95"/>
            <line class="edge" data-link="r2-r3" x1="500" y1="115" x2="500" y2="255"/>
            <line class="edge" data-link="r1-r3" x1="270" y1="196" x2="475" y2="265"/>
            <line class="external-edge" data-external-link="r1-r2" x1="270" y1="154" x2="475" y2="85"/>
            <line class="external-edge" data-external-link="r2-r3" x1="510" y1="115" x2="510" y2="255"/>
            <line class="external-edge" data-external-link="r1-r3" x1="270" y1="206" x2="475" y2="275"/>
            <line class="edge active" data-link="h1-r1" x1="118" y1="180" x2="232" y2="180"/>
            <line class="edge active" data-link="r3-h2" x1="525" y1="275" x2="644" y2="193"/>
            <line class="external-edge" data-external-link="h3-r2" x1="626" y1="55" x2="535" y2="75"/>
            <text class="edge-label" x="377" y="115">r1–r2 <tspan class="edge-value" data-value="r1-r2"></tspan></text>
            <text class="edge-label" x="535" y="188">r2–r3 <tspan class="edge-value" data-value="r2-r3"></tspan></text>
            <text class="edge-label" x="378" y="255">r1–r3 <tspan class="edge-value" data-value="r1-r3"></tspan></text>
            <text class="edge-label" x="174" y="164">host access</text>
            <text class="edge-label" x="608" y="245">host access</text>
            <text class="edge-label" x="588" y="42">external traffic</text>
            <rect class="node host" x="35" y="151" width="84" height="58" rx="14"/>
            <text class="node-name" x="77" y="176">h1</text><text class="node-kind" x="77" y="194">source host</text>
            <circle class="node" cx="257" cy="180" r="35"/><text class="node-name" x="257" y="177">r1</text><text class="node-kind" x="257" y="195">router</text>
            <circle class="node" cx="500" cy="80" r="35"/><text class="node-name" x="500" y="77">r2</text><text class="node-kind" x="500" y="95">router</text>
            <circle class="node" cx="500" cy="280" r="35"/><text class="node-name" x="500" y="277">r3</text><text class="node-kind" x="500" y="295">router</text>
            <rect class="node host" x="644" y="151" width="84" height="58" rx="14"/>
            <text class="node-name" x="686" y="176">h2</text><text class="node-kind" x="686" y="194">destination</text>
            <rect class="node host" x="628" y="16" width="100" height="58" rx="14"/>
            <text class="node-name" x="678" y="41">h3</text><text class="node-kind" x="678" y="59">external sender</text>
          </svg>
        </div>
        <div class="legend"><span><i class="dot"></i>Managed h1 route</span><span style="color:#d5a6ff">Dashed purple = external h3 route</span><span style="color:var(--red)">Red link = high utilization</span></div>
      </article>
      <article class="panel">
        <h2>Recent measured utilization</h2>
        <canvas id="chart" aria-label="Line chart of link utilization over time"></canvas>
        <div class="chart-legend"><span><i class="chart-line"></i>Congested primary link</span><span><i class="chart-line orange"></i>Backup link</span></div>
      </article>
    </div>
    <div>
      <article class="panel">
        <h2>External traffic control</h2>
        <p>h3 is attached to r2. Its UDP traffic to h2 shares the r2–r3 bottleneck with the managed h1 → h2 service.</p>
        <label for="external-rate">Offered external load from h3</label>
        <div class="slider-row"><input id="external-rate" type="range" min="0" max="2" step="0.1" value="0"><output class="rate-value" id="external-rate-value">0.0 Mbps</output></div>
        <div class="controls"><button id="apply-rate">Apply external load</button><span class="status" id="external-status">External flow stopped</span></div>
        <p class="small">Increase the load gradually. If congestion is detected, OSPF should move both h1 and h3 away from r2–r3. Set the slider to 0 to stop external traffic and demonstrate recovery.</p>
        <div class="event"><strong>Managed service flow</strong><p id="protected-status">Starting low-rate h1 → h2 baseline traffic…</p></div>
        <div class="controls"><button class="stop" id="stop-traffic">Stop managed flow</button><button id="start-traffic">Start managed flow</button></div>
      </article>
      <article class="panel">
        <h2>Measured links</h2><div class="links" id="links"><p>Waiting for telemetry…</p></div>
      </article>
      <article class="panel">
        <h2>What is happening?</h2>
        <div class="explain" id="explanation">Waiting for the controller's first model evaluation.</div>
        <div class="event"><strong id="last-action">No decision yet</strong><p id="last-reason">The controller needs enough live samples to build its features.</p><p class="small" id="last-verification"></p></div>
        <p class="small" id="updated">Dashboard is waiting for the lab.</p>
      </article>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const colors = {primary: "#4fd1e5", backup: "#ffb454", grid: "#31435b", text: "#9aabc2"};
let state = null;
function pct(value) { return value == null || value === "" || !Number.isFinite(Number(value)) ? "—" : `${Number(value).toFixed(1)}%`; }
function setText(id, value) { $(id).textContent = value; }
function drawChart(data) {
  const canvas = $("chart"), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  if (!rect.width) return;
  canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, pad = {l: 38, r: 12, t: 12, b: 24};
  ctx.clearRect(0, 0, w, h); ctx.font = "11px system-ui"; ctx.lineWidth = 1;
  for (let v = 0; v <= 100; v += 25) {
    const y = pad.t + (h-pad.t-pad.b) * (1-v/100);
    ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillStyle = colors.text; ctx.fillText(`${v}%`, 2, y+4);
  }
  [[data.primary_link,colors.primary],[data.backup_link,colors.backup]].forEach(([id,color]) => {
    const rows = (data.links[id] || []).slice(-60); if (!rows.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
    rows.forEach((row,i) => {
      const x = pad.l + (w-pad.l-pad.r) * (rows.length < 2 ? 1 : i/(rows.length-1));
      const y = pad.t + (h-pad.t-pad.b) * (1-Math.max(0,Math.min(100,Number(row.utilization_pct)||0))/100);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  });
}
function render(data) {
  state = data;
  $("connection").textContent = `LIVE · updated ${new Date(data.timestamp*1000).toLocaleTimeString()}`;
  $("connection").classList.remove("error");
  const status = data.controller;
  setText("mode", data.mode);
  setText("controller-state", status.running ? (status.last_action_status || "Controller running") : (status.error || "Controller stopped"));
  setText("probability", status.last_prediction == null ? "Warming up" : `${(status.last_prediction*100).toFixed(1)}%`);
  setText("threshold", `Reroute ≥ ${Math.round(data.thresholds.high*100)}% · recover ≤ ${Math.round(data.thresholds.low*100)}% · safeguard ≥ ${Math.round(data.thresholds.reactive)}%`);
  const primarySamples = data.links[data.primary_link] || [];
  const latestPrimary = primarySamples[primarySamples.length-1];
  const currentUtilization = status.last_utilization ?? (latestPrimary && Number(latestPrimary.utilization_pct));
  setText("utilization", pct(currentUtilization));
  setText("utilization-link", `Live measurement on ${data.primary_link.replace("-", "–")}`);
  const path = data.actual_path || [];
  setText("route", path.length ? path.join(" → ") : "Unavailable");
  const primary = JSON.stringify(path) === JSON.stringify(data.primary_path);
  setText("route-kind", primary ? "Primary route via r2" : (JSON.stringify(path) === JSON.stringify(data.backup_path) ? "Backup route, verified by lab" : "Observed from live route lookup"));
  const active = new Set();
  for (let i=0;i<path.length-1;i++) active.add(`${path[i]}-${path[i+1]}`);
  active.add("h1-r1"); active.add("r3-h2");
  document.querySelectorAll("[data-link]").forEach(el => {
    const id = el.dataset.link, samples = data.links[id] || [], latest = samples[samples.length-1];
    el.classList.toggle("active", active.has(id));
    el.classList.toggle("congested", latest && Number(latest.utilization_pct) >= 70);
  });
  const externalActive = new Set();
  const externalPath = data.external_path || [];
  for (let i=0;i<externalPath.length-1;i++) externalActive.add(`${externalPath[i]}-${externalPath[i+1]}`);
  if (data.traffic_running) externalActive.add("h3-r2");
  document.querySelectorAll("[data-external-link]").forEach(el => {
    el.classList.toggle("active", externalActive.has(el.dataset.externalLink));
  });
  Object.keys(data.links).forEach(id => {
    const rows = data.links[id] || [], latest = rows[rows.length-1], value = document.querySelector(`[data-value="${id}"]`);
    if (value) value.textContent = latest ? ` ${pct(latest.utilization_pct)}` : " —";
  });
  $("links").innerHTML = "";
  Object.keys(data.links).forEach(id => {
    const rows = data.links[id] || [], latest = rows[rows.length-1];
    const util = latest ? Math.max(0,Math.min(100,Number(latest.utilization_pct)||0)) : 0;
    const row = document.createElement("div"); row.className = "link-row";
    const head = document.createElement("div"); head.className = "link-head";
    const name = document.createElement("span"); name.textContent = id.replace("-", " ↔ ");
    const val = document.createElement("span"); val.textContent = latest ? pct(util) : "Waiting";
    head.append(name,val);
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("i"); fill.className = `bar-fill${util >= 70 ? " high" : ""}`; fill.style.width = `${util}%`; bar.append(fill);
    const meta = document.createElement("div"); meta.className="link-meta"; meta.textContent = latest ? `packets/s ${Number(latest.packets_per_sec||0).toFixed(1)} · drops ${latest.dropped_delta||0} · ${rows.length} recent samples` : "No samples collected yet";
    row.append(head,bar,meta); $("links").append(row);
  });
  drawChart(data);
  const decision = status.last_decision;
  setText("last-action", decision ? `${decision.action} · ${decision.actuation_status || status.last_action_status || "evaluated"}` : "Collecting warm-up samples");
  setText("last-reason", decision ? decision.reason : "The controller needs enough live samples to build its features.");
  setText("last-verification", decision ? (decision.verification || (decision.should_execute ? "Awaiting route verification." : "")) : "");
  const mode = data.mode === "ACTUATION ENABLED" ? "When predicted risk crosses the high threshold (or the reactive utilization safeguard triggers), the controller raises the r2–r3 OSPF cost and verifies the h1 route. OSPF then also changes h3's route. After the offered load falls, hysteresis and hold time prevent flapping before the primary route is restored." : "Dry-run computes the same prediction and decision but never changes OSPF costs. Both highlighted paths are read from live Mininet route lookups.";
  setText("explanation", `${mode} Bottleneck ${data.primary_link}: ${pct(currentUtilization)} utilization; congestion probability ${status.last_prediction == null ? "is warming up" : `${(status.last_prediction*100).toFixed(1)}%`}. Managed path: ${path.length ? path.join(" → ") : "not available"}. External path: ${externalPath.length ? externalPath.join(" → ") : "not available"}.`);
  const rate = Number(data.external_rate_mbps || 0);
  if (document.activeElement !== $("external-rate")) $("external-rate").value = rate;
  setText("external-rate-value", `${rate.toFixed(1)} Mbps`);
  setText("external-status", data.traffic_running ? `h3 sending at ${rate.toFixed(1)} Mbps` : (rate > 0 ? "External flow is not running; inspect the lab terminal." : "External flow stopped"));
  $("start-traffic").disabled = data.protected_running;
  $("stop-traffic").disabled = !data.protected_running;
  setText("protected-status", data.protected_running ? "Active: h1 → h2 at 0.4 Mbps, sharing the measured bottleneck." : "Stopped. Start it to observe the managed flow move to the backup path.");
  setText("updated", `Last refreshed ${new Date(data.timestamp*1000).toLocaleTimeString()} · auto-refresh every second.`);
}
async function refresh() {
  try {
    const response = await fetch("/api/state", {cache:"no-store"});
    if (!response.ok) throw new Error(`Dashboard API returned HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("connection").textContent = `NOT CONNECTED · ${error.message}`;
    $("connection").classList.add("error");
  }
}
async function traffic(action) {
  $("start-traffic").disabled = true; $("stop-traffic").disabled = true;
  setText("protected-status", action === "start" ? "Starting managed flow…" : "Stopping managed flow…");
  try {
    const response = await fetch("/api/traffic", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    setText("protected-status", result.message);
  } catch (error) { setText("protected-status", error.message); }
  await refresh();
}
async function setExternalRate() {
  const rate = Number($("external-rate").value);
  $("apply-rate").disabled = true;
  setText("external-status", rate === 0 ? "Stopping external flow…" : `Setting external load to ${rate.toFixed(1)} Mbps…`);
  try {
    const response = await fetch("/api/traffic", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:"set_external_rate",rate_mbps:rate})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    setText("external-status", result.message);
  } catch (error) { setText("external-status", error.message); }
  $("apply-rate").disabled = false;
  await refresh();
}
$("external-rate").addEventListener("input", () => setText("external-rate-value", `${Number($("external-rate").value).toFixed(1)} Mbps`));
$("apply-rate").addEventListener("click", setExternalRate);
$("start-traffic").addEventListener("click", () => traffic("start"));
$("stop-traffic").addEventListener("click", () => traffic("stop"));
window.addEventListener("resize", () => { if (state) drawChart(state); });
refresh(); setInterval(refresh, 1000);
</script>
</body>
</html>
"""


class LiveDashboard:
    """Serve a local live view of a caller-owned Mininet controller."""

    def __init__(
        self,
        snapshot_provider: Callable[[], dict[str, Any]],
        traffic_handler: Callable[[str, float | None], str],
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("The live dashboard must bind to a loopback address.")
        if not 0 <= port <= 65535:
            raise ValueError("Dashboard port must be between 0 and 65535.")
        self.snapshot_provider = snapshot_provider
        self.traffic_handler = traffic_handler
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Start the dashboard before requesting its URL.")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Dashboard is already running.")
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if self.path == "/api/state":
                    try:
                        payload = json.dumps(
                            dashboard.snapshot_provider(),
                            allow_nan=False,
                        ).encode("utf-8")
                    except Exception as exc:
                        self._json(500, {"error": str(exc)})
                        return
                    self._send(200, payload, "application/json; charset=utf-8")
                    return
                self._json(404, {"error": "Not found."})

            def do_POST(self) -> None:
                if self.path != "/api/traffic":
                    self._json(404, {"error": "Not found."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 4096:
                        raise ValueError("Invalid request body size.")
                    payload = json.loads(self.rfile.read(length))
                    action = payload.get("action") if isinstance(payload, dict) else None
                    if action not in {"start", "stop", "set_external_rate"}:
                        raise ValueError("Unsupported traffic action.")
                    rate = None
                    if action == "set_external_rate":
                        rate = payload.get("rate_mbps")
                        if (
                            isinstance(rate, bool)
                            or not isinstance(rate, (int, float))
                            or not math.isfinite(rate)
                            or not 0 <= rate <= 2
                        ):
                            raise ValueError(
                                "External offered load must be between 0 and 2 Mbps."
                            )
                    message = dashboard.traffic_handler(action, rate)
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                except Exception as exc:
                    self._json(500, {"error": str(exc)})
                    return
                self._json(200, {"message": message})

            def _json(self, status: int, value: dict[str, Any]) -> None:
                self._send(
                    status,
                    json.dumps(value).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="live-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None
