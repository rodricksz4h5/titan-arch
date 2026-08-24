// Titan Arch -- dashboard front-end
const $ = (s) => document.querySelector(s);

let map, edgeLayer, nodeLayer, rerouteLayer;
let GRAPH = null;           // last /api/graph payload
let CLOSED = [];            // currently closed junction ids
let NODE_INDEX = {};        // id -> {lat, lon, criticality}

// colour a road/junction by criticality (0..1-ish, we rescale to the max seen)
function critColor(c, cmax) {
  const t = cmax > 0 ? c / cmax : 0;
  if (t < 0.25) return "#3a7d5c";
  if (t < 0.55) return "#d9a441";
  return "#e2574c";
}
function critWeight(c, cmax) {
  const t = cmax > 0 ? c / cmax : 0;
  return 2 + t * 5;
}

function toast(msg, ms = 2600) {
  const el = $("#toast");
  el.textContent = msg; el.classList.remove("hidden");
  clearTimeout(el._t); el._t = setTimeout(() => el.classList.add("hidden"), ms);
}

async function loadGraph() {
  const r = await fetch("/api/graph");
  GRAPH = await r.json();
  render();
}

function render() {
  if (typeof stopFlow === "function") stopFlow();
  $("#metaName").textContent = GRAPH.meta.name || "Road network";
  $("#nNodes").textContent = GRAPH.n_nodes;
  $("#nEdges").textContent = GRAPH.n_edges;
  if ($("#weightTag")) $("#weightTag").classList.toggle("hidden", !GRAPH.demand_weighted);

  // resilience index
  const ri = GRAPH.resilience_index;
  $("#riValue").textContent = ri.toFixed(2);
  const lab = ri > 0.7 ? "robust" : ri > 0.4 ? "exposed" : "fragile";
  const labEl = $("#riLabel");
  labEl.textContent = lab;
  labEl.style.color = ri > 0.7 ? "#5ad19a" : ri > 0.4 ? "#d9a441" : "#e2574c";
  if ($("#riApl")) $("#riApl").textContent =
    GRAPH.resilience_index_apl != null ? GRAPH.resilience_index_apl.toFixed(2) : "—";

  // node index + max criticality
  NODE_INDEX = {};
  let cmax = 0;
  GRAPH.geojson.nodes.features.forEach((f) => {
    const c = f.properties.criticality;
    NODE_INDEX[f.properties.id] = {
      lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], criticality: c,
    };
    if (c > cmax) cmax = c;
  });
  GRAPH._cmax = cmax || 1e-9;

  drawMap();
  drawGatekeepers();
  drawStressCurve();
  drawReconstruction();
  drawEmergency();
  resetCongestion();
  resetFloodPanel();
}

let facilityLayer = null;
function drawEmergency() {
  const em = GRAPH.emergency;
  const block = $("#emergencyBlock");
  if (facilityLayer) { map.removeLayer(facilityLayer); facilityLayer = null; }
  if (!em) { block.classList.add("hidden"); return; }
  block.classList.remove("hidden");

  const served = em.served_fraction != null ? em.served_fraction * 100 : null;
  $("#emFacilities").textContent = em.n_facilities;
  const sEl = $("#emServed");
  sEl.textContent = served != null ? served.toFixed(0) + "%" : "—";
  sEl.classList.toggle("danger", served != null && served < 80);

  const mx = Math.max(...em.critical.map((c) => c.score), 1e-9);
  $("#emList").innerHTML = em.critical.map((c, i) =>
    `<li class="gk-item" data-lat="${c.lat}" data-lon="${c.lon}">
       <span class="gk-rank">${i + 1}</span>
       <div class="gk-bar"><i style="width:${(100 * c.score / mx).toFixed(0)}%;background:linear-gradient(90deg,#ff8a5a,#ff5a5a)"></i></div>
       <span class="gk-score">${c.score.toFixed(2)}</span></li>`).join("");
  $("#emList").querySelectorAll(".gk-item").forEach((li) =>
    li.onclick = () => map.setView([+li.dataset.lat, +li.dataset.lon], 18, { animate: true }));

  // facility markers
  facilityLayer = L.layerGroup();
  em.facilities.forEach((f) => {
    L.marker([f.lat, f.lon], { icon: L.divIcon({
      className: "", html: '<span class="fac-cross">✚</span>', iconSize: [14, 14] }) })
      .bindTooltip(f.kind, { direction: "top" }).addTo(facilityLayer);
  });
  facilityLayer.addTo(map);
}

function resetFloodPanel() {
  const isCity = GRAPH.meta && GRAPH.meta.source === "OpenStreetMap";
  $("#floodBlock").classList.toggle("hidden", !isCity);
  $("#waveBlock").classList.toggle("hidden", !isCity);
  if (waveLayer) { map.removeLayer(waveLayer); waveLayer = null; }
  $("#waveStat").innerHTML = "";
  const wb = $("#waveBtn"); wb.disabled = false; wb.textContent = "▶ Play access wave";
  FLOOD = null;
  if (floodLayer) { map.removeLayer(floodLayer); floodLayer = null; }
  $("#floodCtl").classList.add("hidden");
  const b = $("#floodPrepBtn");
  b.classList.remove("hidden"); b.disabled = false; b.textContent = "Prepare flood model";
}

function drawReconstruction() {
  const block = $("#reconBlock");
  const rec = GRAPH.reconstruction;
  if (!rec) { block.classList.add("hidden"); return; }
  block.classList.remove("hidden");
  const b = rec.before, a = rec.after;
  const row = (lbl, raw, heal, good) =>
    `<span class="lbl">${lbl}</span><span class="raw">${raw}</span><span class="heal">${heal}</span>`;
  $("#reconStats").innerHTML =
    `<span class="h">metric</span><span class="h raw">raw</span><span class="h heal">healed</span>` +
    row("disconnected pieces", b.components, a.components) +
    row("largest area", (b.lcc_frac * 100).toFixed(0) + "%", (a.lcc_frac * 100).toFixed(0) + "%") +
    row("roads (edges)", b.edges, a.edges) +
    `<span class="lbl">bridges added</span><span class="raw"></span><span class="heal">${rec.bridges_added}</span>`;

  // confidence read-out + replay availability
  const healed = (GRAPH.geojson.edges.features || []).filter((f) => f.properties.healed);
  if (healed.length) {
    const rows = healed.map((f) => ({ cf: healConfidence(f.properties.length || 0) }));
    renderHealConf(rows, rec);
    $("#healReplayBtn").classList.remove("hidden");
  } else {
    $("#healConf").classList.add("hidden");
    $("#healReplayBtn").classList.add("hidden");
  }
}

function drawStressCurve() {
  const svg = $("#stressChart");
  const W = 280, H = 70, pad = 4;
  const pts = [{ eff_ratio: 1.0 }].concat(GRAPH.stress_curve || []);
  if (pts.length < 2) { svg.innerHTML = ""; return; }
  const n = pts.length;
  const x = (i) => pad + (W - 2 * pad) * (i / (n - 1));
  const y = (v) => pad + (H - 2 * pad) * (1 - Math.max(0, Math.min(1, v)));
  let dLine = "", dArea = `M ${x(0)} ${y(0)} `;
  pts.forEach((p, i) => {
    dLine += (i ? "L" : "M") + ` ${x(i).toFixed(1)} ${y(p.eff_ratio).toFixed(1)} `;
    dArea += `L ${x(i).toFixed(1)} ${y(p.eff_ratio).toFixed(1)} `;
  });
  dArea += `L ${x(n - 1)} ${y(0)} Z`;
  svg.innerHTML =
    `<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
       <stop offset="0%" stop-color="#e2574c" stop-opacity="0.35"/>
       <stop offset="100%" stop-color="#e2574c" stop-opacity="0.02"/></linearGradient></defs>
     <line x1="${pad}" y1="${y(0)}" x2="${W - pad}" y2="${y(0)}" stroke="#28303c" stroke-width="1"/>
     <path d="${dArea}" fill="url(#sg)"/>
     <path d="${dLine}" fill="none" stroke="#e2574c" stroke-width="2"/>
     ${pts.map((p, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(p.eff_ratio).toFixed(1)}" r="2" fill="#e2574c"/>`).join("")}`;
}

function drawMap() {
  const c = GRAPH.meta.center || [12.9695, 77.593];
  if (!map) {
    map = L.map("map", { zoomControl: true }).setView(c, 16);
    const dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO", maxZoom: 20,
    }).addTo(map);
    const sat = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Imagery &copy; Esri", maxZoom: 19 });
    L.control.layers({ "Dark map": dark, "Satellite": sat }, null,
      { position: "topright", collapsed: false }).addTo(map);
  }
  [edgeLayer, nodeLayer, rerouteLayer].forEach((l) => l && map.removeLayer(l));

  const cmax = GRAPH._cmax;
  const showHealed = $("#showHealed") ? $("#showHealed").checked : true;
  edgeLayer = L.geoJSON(GRAPH.geojson.edges, {
    filter: (f) => showHealed || !f.properties.healed,
    style: (f) => {
      if (CONGESTION_MODE && CONG_VC) {
        const { u, v } = f.properties;
        const key = u < v ? `${u}|${v}` : `${v}|${u}`;
        return { color: vcColor(CONG_VC[key]), weight: 3.2, opacity: 0.92 };
      }
      return f.properties.healed
        ? { color: "#5ad19a", weight: 4, opacity: 0.95, dashArray: "5 5" }
        : { color: critColor(f.properties.criticality, cmax),
            weight: critWeight(f.properties.criticality, cmax), opacity: 0.9 };
    },
  }).addTo(map);

  nodeLayer = L.geoJSON(GRAPH.geojson.nodes, {
    pointToLayer: (f, latlng) => {
      const id = f.properties.id;
      const closed = CLOSED.includes(id);
      const c = f.properties.criticality;
      return L.circleMarker(latlng, {
        radius: closed ? 7 : 3 + 6 * (c / cmax),
        color: closed ? "#e2574c" : "#0e1116",
        weight: closed ? 2 : 1,
        fillColor: closed ? "#e2574c" : critColor(c, cmax),
        fillOpacity: closed ? 0.95 : 0.85,
      });
    },
    onEachFeature: (f, layer) => {
      const id = f.properties.id;
      layer.on("click", () => toggleClose(id));
      layer.bindTooltip(`junction ${id}`, { direction: "top" });
    },
  }).addTo(map);

  // fit bounds once
  try { map.fitBounds(edgeLayer.getBounds(), { padding: [30, 30] }); } catch (e) {}
}

function drawGatekeepers() {
  const cmax = GRAPH._cmax;
  const ol = $("#gkList");
  ol.innerHTML = "";
  GRAPH.gatekeepers.forEach((g, i) => {
    const li = document.createElement("li");
    li.className = "gk-item";
    li.innerHTML = `<span class="gk-rank">${i + 1}</span>
      <div class="gk-bar"><i style="width:${(100 * g.criticality / cmax).toFixed(0)}%"></i></div>
      <span class="gk-score">${g.criticality.toFixed(3)}</span>`;
    li.onclick = () => {
      map.setView([g.lat, g.lon], 18, { animate: true });
      const dem = g.demand != null ? ` · demand ${g.demand}` : "";
      toast(`Junction ${g.id}: ${(100 * g.criticality / cmax).toFixed(0)}% of the city's worst-case load${dem}`);
    };
    ol.appendChild(li);
  });
}

async function toggleClose(id) {
  const i = CLOSED.indexOf(id);
  if (i >= 0) CLOSED.splice(i, 1); else CLOSED.push(id);
  await runAblation();
}

async function runAblation() {
  if (CLOSED.length === 0) {
    $("#simResult").classList.add("hidden");
    $("#resetBtn").classList.add("hidden");
    $("#simHelp").classList.remove("hidden");
    if (rerouteLayer) map.removeLayer(rerouteLayer);
    if (CONGESTION_MODE && GRAPH.congestion) CONG_VC = GRAPH.congestion.vc;
    $("#congStat").innerHTML = "";
    drawMap();
    return;
  }
  // probe a cross-city trip between two important junctions that are NOT closed
  const open = GRAPH.gatekeepers.filter((g) => !CLOSED.includes(g.id));
  const pair = open.length >= 2 ? [open[0].id, open[open.length - 1].id] : [];
  const r = await fetch("/api/ablate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dead: CLOSED, pairs: pair.length ? [pair] : [],
      congestion: CONGESTION_MODE }),
  });
  const out = await r.json();
  if (CONGESTION_MODE && out.congestion) {
    CONG_VC = out.congestion.vc;
    const c = out.congestion;
    $("#congStat").innerHTML =
      `<span class="k">extra traffic delay</span><span class="v danger">+${c.extra_delay_pct.toFixed(0)}%</span>` +
      `<span class="k">road-km over capacity</span><span class="v">${c.base_overloaded.toFixed(0)}% → ${c.new_overloaded.toFixed(0)}%</span>`;
  }
  showAblation(out);
  drawMap();
  drawReroute(out);
}

function drawReroute(out) {
  if (rerouteLayer) map.removeLayer(rerouteLayer);
  const t = (out.trips || [])[0];
  if (t && t.new_path_coords) {
    rerouteLayer = L.geoJSON(
      { type: "Feature", geometry: { type: "LineString",
        coordinates: t.new_path_coords } },
      { style: { color: "#7aa2ff", weight: 5, opacity: 0.95, dashArray: "1 8" } }
    ).addTo(map);
  }
}

function showAblation(out) {
  $("#simHelp").classList.add("hidden");
  const box = $("#simResult");
  box.classList.remove("hidden");
  $("#resetBtn").classList.remove("hidden");

  const effDrop = (1 - out.efficiency_ratio) * 100;
  const lcc = (out.lcc_frac * 100).toFixed(0);
  let html = `<div><span class="big">${CLOSED.length}</span> junction(s) closed</div>
    <div>City-wide efficiency: <span class="drop">−${effDrop.toFixed(1)}%</span></div>
    <div>Largest connected area now: <b>${lcc}%</b> of the city</div>`;

  const t = (out.trips || [])[0];
  if (t) {
    if (t.status === "cut_off") {
      html += `<div class="cut">⚠ A cross-city trip is now completely cut off.</div>`;
    } else if (t.status === "rerouted" && t.delta_m != null) {
      const mins = t.delta_min != null ? ` · <b>+${t.delta_min.toFixed(1)} min</b>` : "";
      html += `<div>Sample cross-city trip: <b>+${Math.round(t.delta_m)} m</b>${mins} detour.</div>`;
    }
  }
  box.innerHTML = html;
}

$("#resetBtn").onclick = () => { CLOSED = []; runAblation(); };

let repairLayer = null;
async function suggestRepairs() {
  const btn = $("#repairBtn");
  btn.disabled = true; btn.textContent = "Analysing…";
  toast("Searching for the highest-impact new link…", 9000);
  try {
    const r = await fetch("/api/recommend_repairs", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: "{}" });
    const data = await r.json();
    if (data.error) { toast("⚠ " + data.error, 4000); return; }
    showRepairs(data);
  } catch (e) {
    toast("Request failed: " + e.message, 4000);
  } finally {
    btn.disabled = false; btn.textContent = "Suggest the best fix";
  }
}

function showRepairs(data) {
  const box = $("#repairResult");
  box.classList.remove("hidden");
  const s = data.suggestions || [];
  if (!s.length) { box.innerHTML = "No useful new link found (city is already well-knit)."; return; }
  const before = data.resilience_before, after = data.resilience_after;
  let html = "";
  if (before != null && after != null) {
    html += `<div class="ri-jump">Resilience ${before.toFixed(2)}
      <span class="arrow">→</span> ${after.toFixed(2)}</div>
      <div>by adding one ${Math.round(s[0].length_m)} m link.</div>`;
  }
  html += `<ul class="repair-list">` + s.map((x, i) =>
    `<li class="repair-item" data-i="${i}">
       <span>#${i + 1} ${x.reconnects ? "reconnects a cut-off area" : "cuts a big detour"}</span>
       <span class="reco-len">${Math.round(x.length_m)} m · <span class="gain">+${x.eff_gain_pct}%</span></span>
     </li>`).join("") + `</ul>`;
  box.innerHTML = html;

  drawRepairs(s);
  box.querySelectorAll(".repair-item").forEach((li) => {
    li.onclick = () => {
      const x = s[+li.dataset.i];
      map.fitBounds([[x.u_lat, x.u_lon], [x.v_lat, x.v_lon]], { padding: [80, 80] });
    };
  });
}

function drawRepairs(s) {
  if (repairLayer) map.removeLayer(repairLayer);
  repairLayer = L.layerGroup();
  s.forEach((x, i) => {
    L.polyline([[x.u_lat, x.u_lon], [x.v_lat, x.v_lon]], {
      color: "#5ad19a", weight: i === 0 ? 6 : 3,
      opacity: i === 0 ? 0.95 : 0.6, dashArray: "6 6",
    }).addTo(repairLayer);
  });
  repairLayer.addTo(map);
}

$("#repairBtn").onclick = suggestRepairs;

// ---- animated cascading failure (the node-ablation stress test, visualised) ----
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ADJ = null;

function buildAdjacency() {
  ADJ = new Map();
  GRAPH.geojson.edges.features.forEach((f) => {
    const { u, v } = f.properties;
    if (!ADJ.has(u)) ADJ.set(u, []);
    if (!ADJ.has(v)) ADJ.set(v, []);
    ADJ.get(u).push(v); ADJ.get(v).push(u);
  });
}

function largestComponent(removed) {
  const seen = new Set(removed);
  let best = new Set();
  for (const start of ADJ.keys()) {
    if (seen.has(start)) continue;
    const comp = [start], q = [start]; seen.add(start);
    while (q.length) {
      const n = q.pop();
      for (const nb of ADJ.get(n) || []) if (!seen.has(nb)) { seen.add(nb); q.push(nb); comp.push(nb); }
    }
    if (comp.length > best.size) best = new Set(comp);
  }
  return best;
}

function renderCascade(removed, largest) {
  const cmax = GRAPH._cmax;
  [edgeLayer, nodeLayer, rerouteLayer, floodLayer, facilityLayer, repairLayer]
    .forEach((l) => l && map.removeLayer(l));
  edgeLayer = L.geoJSON(GRAPH.geojson.edges, {
    style: (f) => {
      const { u, v } = f.properties;
      if (removed.has(u) || removed.has(v)) return { color: "#222", weight: 1, opacity: 0.15 };
      if (largest.has(u) && largest.has(v))
        return { color: critColor(f.properties.criticality, cmax),
                 weight: critWeight(f.properties.criticality, cmax), opacity: 0.9 };
      return { color: "#e2904c", weight: 2, opacity: 0.85 };   // cut-off district
    },
  }).addTo(map);
  nodeLayer = L.geoJSON(GRAPH.geojson.nodes, {
    pointToLayer: (f, ll) => {
      const id = f.properties.id;
      if (removed.has(id))
        return L.circleMarker(ll, { radius: 7, color: "#fff", weight: 2,
          fillColor: "#e2574c", fillOpacity: 1 });
      if (!largest.has(id))
        return L.circleMarker(ll, { radius: 4, color: "#e2904c", weight: 1,
          fillColor: "#e2904c", fillOpacity: 0.8 });
      return L.circleMarker(ll, { radius: 2.5, color: "#0e1116", weight: 1,
        fillColor: "#6db8ff", fillOpacity: 0.5 });
    },
  }).addTo(map);
}

// draw a curve into #stressChart progressively (largest-connected fraction 0..1)
function drawLiveStress(points) {
  const svg = $("#stressChart");
  const W = 280, H = 70, pad = 4;
  const N = (GRAPH.stress_order || []).length + 1;   // fixed x-axis length
  if (N < 2) return;
  const x = (i) => pad + (W - 2 * pad) * (i / (N - 1));
  const y = (v) => pad + (H - 2 * pad) * (1 - Math.max(0, Math.min(1, v)));
  let dLine = "", dArea = `M ${x(0)} ${y(0)} `;
  points.forEach((v, i) => {
    dLine += (i ? "L" : "M") + ` ${x(i).toFixed(1)} ${y(v).toFixed(1)} `;
    dArea += `L ${x(i).toFixed(1)} ${y(v).toFixed(1)} `;
  });
  const last = points.length - 1;
  dArea += `L ${x(last)} ${y(0)} Z`;
  svg.innerHTML =
    `<defs><linearGradient id="sgl" x1="0" y1="0" x2="0" y2="1">
       <stop offset="0%" stop-color="#e2574c" stop-opacity="0.4"/>
       <stop offset="100%" stop-color="#e2574c" stop-opacity="0.02"/></linearGradient></defs>
     <line x1="${pad}" y1="${y(0)}" x2="${W - pad}" y2="${y(0)}" stroke="#28303c" stroke-width="1"/>
     <path d="${dArea}" fill="url(#sgl)"/>
     <path d="${dLine}" fill="none" stroke="#e2574c" stroke-width="2"/>
     <circle cx="${x(last).toFixed(1)}" cy="${y(points[last]).toFixed(1)}" r="3.2" fill="#fff"
       stroke="#e2574c" stroke-width="2"/>`;
}

async function runCascade() {
  const order = GRAPH.stress_order || [];
  if (!order.length) { toast("Load a city first."); return; }
  buildAdjacency();
  CASCADE_FAILED = new Set();
  if (!FLOW || !FLOW.active) startFlow();        // particles flow alongside the test
  const btn = $("#cascadeBtn"); btn.disabled = true; btn.textContent = "Running…";
  $("#cascadeReadout").classList.remove("hidden");
  $("#stressCaption").textContent = "↑ live: % of the city still connected →";
  const removed = new Set();
  const base = largestComponent(removed).size || 1;
  const live = [1.0];
  drawLiveStress(live);
  for (let i = 0; i < order.length; i++) {
    removed.add(order[i]);
    CASCADE_FAILED.add(String(order[i]));        // traffic on this junction re-routes live
    const largest = largestComponent(removed);
    const g = GRAPH.gatekeepers.find((x) => x.id === order[i]);
    if (g) map.panTo([g.lat, g.lon], { animate: true, duration: 0.4 });
    renderCascade(removed, largest);
    const frac = largest.size / base;
    live.push(frac);
    drawLiveStress(live);
    const pct = 100 * frac;
    $("#cascadeReadout").innerHTML =
      `<div><span class="ri-now">${i + 1}</span> gatekeepers failed</div>
       <div>largest connected area: <b>${pct.toFixed(0)}%</b> of the city</div>
       <div class="lost">${(100 - pct).toFixed(0)}% cut off / fragmented</div>`;
    await sleep(700);      // brief dwell so the traffic re-routes between failures
  }
  btn.disabled = false; btn.textContent = "▶ Replay stress test (live curve)";
  await sleep(2200);
  $("#stressCaption").textContent = "city efficiency as gatekeepers fail →";
  CASCADE_FAILED = new Set();                     // traffic recovers across the full network
  drawStressCurve();
  drawMap();                                      // restore the map; particles keep floating
}

$("#cascadeBtn").onclick = runCascade;

// ---- reachability wavefront (animated isochrone from a hospital) ----
class MinHeap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(x) { const a = this.a; a.push(x); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]]; i = p; } }
  pop() { const a = this.a, top = a[0], last = a.pop();
    if (a.length) { a[0] = last; let i = 0; for (;;) { let l = 2*i+1, r = l+2-1, s = i;
      if (l < a.length && a[l][0] < a[s][0]) s = l;
      if (r < a.length && a[r][0] < a[s][0]) s = r;
      if (s === i) break; [a[s], a[i]] = [a[i], a[s]]; i = s; } } return top; }
}

let WADJ = null;
const waveCanvas = L.canvas({ padding: 0.5 });
let waveLayer = null;

function buildWeightedAdj() {
  WADJ = new Map();
  GRAPH.geojson.edges.features.forEach((f) => {
    const { u, v, length } = f.properties;
    const w = length || 1;
    if (!WADJ.has(u)) WADJ.set(u, []);
    if (!WADJ.has(v)) WADJ.set(v, []);
    WADJ.get(u).push([v, w]); WADJ.get(v).push([u, w]);
  });
}

function nearestNode(lat, lon) {
  let best = null, bd = Infinity;
  for (const id in NODE_INDEX) {
    const nd = NODE_INDEX[id]; const d = (nd.lat-lat)**2 + (nd.lon-lon)**2;
    if (d < bd) { bd = d; best = id; }
  }
  return best;
}

function dijkstra(src, removed) {
  const dist = new Map([[src, 0]]);
  const h = new MinHeap(); h.push([0, src]);
  while (h.size) {
    const [d, u] = h.pop();
    if (d > (dist.get(u) ?? Infinity)) continue;
    for (const [v, w] of WADJ.get(u) || []) {
      if (removed.has(v)) continue;
      const nd = d + w;
      if (nd < (dist.get(v) ?? Infinity)) { dist.set(v, nd); h.push([nd, v]); }
    }
  }
  return dist;   // metres
}

function isoColor(min) {
  if (min == null) return "#2a2a2a";
  if (min < 3) return "#2ecc71";
  if (min < 6) return "#9bd84a";
  if (min < 10) return "#e8d24a";
  if (min < 15) return "#e8a14a";
  return "#e2574c";
}

function pickWaveSource() {
  let lat, lon, label;
  if (GRAPH.emergency && GRAPH.emergency.facilities.length) {
    const f = GRAPH.emergency.facilities[0]; lat = f.lat; lon = f.lon; label = "a hospital";
  } else if (GRAPH.gatekeepers.length) {
    const g = GRAPH.gatekeepers[0]; lat = g.lat; lon = g.lon; label = "the central junction";
  } else return null;
  const id = nearestNode(lat, lon);
  return { id, lat: NODE_INDEX[id].lat, lon: NODE_INDEX[id].lon, label };
}

const M_PER_MIN = 30 * 1000 / 60;     // 30 km/h
function toMin(m) { return m == null ? null : m / M_PER_MIN; }

function drawWave(src, distM, T, removed) {
  if (waveLayer) map.removeLayer(waveLayer);
  waveLayer = L.layerGroup();
  for (const id in NODE_INDEX) {
    const nd = NODE_INDEX[id];
    if (removed.has(id)) {
      L.circleMarker([nd.lat, nd.lon], { renderer: waveCanvas, radius: 6, color: "#fff",
        weight: 2, fillColor: "#e2574c", fillOpacity: 1 }).addTo(waveLayer);
      continue;
    }
    const mins = toMin(distM.get(id));
    if (mins == null || mins > T) continue;       // not reached by the wavefront yet
    L.circleMarker([nd.lat, nd.lon], { renderer: waveCanvas, radius: 3.6, stroke: false,
      fillColor: isoColor(mins), fillOpacity: 0.92 }).addTo(waveLayer);
  }
  L.circleMarker([src.lat, src.lon], { renderer: waveCanvas, radius: 8, color: "#fff",
    weight: 2, fillColor: "#7aa2ff", fillOpacity: 1 }).addTo(waveLayer);
  waveLayer.addTo(map);
}

function pctWithin(distM, removed, minutes) {
  const lim = minutes * M_PER_MIN; let n = 0, tot = 0;
  for (const id in NODE_INDEX) {
    if (removed.has(id)) continue;
    tot++; const d = distM.get(id);
    if (d != null && d <= lim) n++;
  }
  return tot ? 100 * n / tot : 0;
}

async function playWave() {
  if (!GRAPH || !GRAPH.gatekeepers || !GRAPH.gatekeepers.length) { toast("Load a city first."); return; }
  buildWeightedAdj();
  const src = pickWaveSource();
  if (!src) return;
  $("#waveSrc").textContent = src.label;
  const fail = $("#waveFail").checked;
  const removed = new Set(fail ? [GRAPH.gatekeepers[0].id] : []);
  const btn = $("#waveBtn"); btn.disabled = true; btn.textContent = "Spreading…";
  if (nodeLayer) map.removeLayer(nodeLayer);

  const distM = dijkstra(src.id, removed);
  // before/after stat (both scenarios)
  const base = pctWithin(dijkstra(src.id, new Set()), new Set(), 10);
  const now = pctWithin(distM, removed, 10);
  const finite = [...distM.values()].map(toMin).filter((v) => v != null);
  const maxT = Math.max(...finite, 1);
  const FRAMES = 34;
  for (let f = 1; f <= FRAMES; f++) {
    drawWave(src, distM, maxT * f / FRAMES, removed);
    await sleep(70);
  }
  $("#waveStat").innerHTML =
    `<span class="k">within 10 min of care (normal)</span><span class="v">${base.toFixed(0)}%</span>` +
    (fail ? `<span class="k">after #1 gatekeeper fails</span><span class="v ${now<base-3?'danger':''}">${now.toFixed(0)}%</span>` : "");
  btn.disabled = false; btn.textContent = "▶ Replay access wave";
}

$("#waveBtn").onclick = playWave;

// ---- Layer 4: traffic congestion view ----
let CONGESTION_MODE = false, CONG_VC = null;
function vcColor(vc) {
  if (vc == null) return "#3a4250";
  if (vc < 0.6) return "#2ecc71";
  if (vc < 0.85) return "#9bd84a";
  if (vc < 1.0) return "#e8d24a";
  if (vc < 1.5) return "#e8a14a";
  return "#e2574c";
}
function resetCongestion() {
  const isCity = GRAPH.meta && GRAPH.meta.source === "OpenStreetMap";
  $("#congBlock").classList.toggle("hidden", !isCity);
  CONGESTION_MODE = false; CONG_VC = null;
  GRAPH.congestion = null;
  if ($("#congToggle")) $("#congToggle").checked = false;
  $("#congStat").innerHTML = "";
  $("#congOverload").textContent = "—";
}
if ($("#congToggle")) $("#congToggle").onchange = async () => {
  CONGESTION_MODE = $("#congToggle").checked;
  if (CONGESTION_MODE && !GRAPH.congestion) {
    toast("Computing peak-hour traffic assignment…", 8000);
    try {
      const r = await fetch("/api/congestion", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: "{}" });
      const c = await r.json();
      if (c.error) { toast("⚠ " + c.error, 4000); $("#congToggle").checked = false; CONGESTION_MODE = false; return; }
      GRAPH.congestion = c;
      $("#congOverload").textContent = c.pct_overloaded.toFixed(0) + "%";
    } catch (e) { toast("Failed: " + e.message, 4000); $("#congToggle").checked = false; CONGESTION_MODE = false; return; }
  }
  CONG_VC = CONGESTION_MODE && GRAPH.congestion ? GRAPH.congestion.vc : null;
  $("#congStat").innerHTML = "";
  drawMap();
  toast(CONGESTION_MODE ? "Congestion heatmap on — fail a junction to watch traffic redistribute."
                        : "Congestion heatmap off.");
};

// ---- Layer 2: flood scenario ----
let FLOOD = null, floodLayer = null;

$("#floodPrepBtn").onclick = async () => {
  const btn = $("#floodPrepBtn");
  btn.disabled = true; btn.textContent = "Reading terrain…";
  toast("Fetching ground elevation for every junction…", 10000);
  try {
    const r = await fetch("/api/flood_prepare", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: "{}" });
    const data = await r.json();
    if (data.error) { toast("⚠ " + data.error, 5000); return; }
    FLOOD = data;
    $("#floodCtl").classList.remove("hidden");
    btn.classList.add("hidden");
    drawFloodChart(data.curve);
    $("#floodSlider").value = 0; $("#floodDepth").textContent = "0.0";
    runFlood(0);
    toast(`Terrain ready · relief ${Math.round(data.max_elev - data.min_elev)} m`);
  } catch (e) { toast("Failed: " + e.message, 5000); }
  finally { btn.disabled = false; }
};

let floodTimer = null;
$("#floodSlider").addEventListener("input", () => {
  if (!FLOOD) return;
  const depth = FLOOD.max_depth * ($("#floodSlider").value / 100);
  $("#floodDepth").textContent = depth.toFixed(1);
  clearTimeout(floodTimer);
  floodTimer = setTimeout(() => runFlood(depth), 120);   // debounce
});

async function runFlood(depth) {
  const r = await fetch("/api/flood", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify({ depth_m: depth }) });
  const sc = await r.json();
  if (sc.error) { toast("⚠ " + sc.error, 4000); return; }
  drawFlooded(sc.flooded);
  const access = sc.access_retained * 100, flooded = sc.frac_junctions_flooded * 100;
  const danger = access < 60 ? "danger" : "";
  let html =
    `<span class="k">junctions flooded</span><span class="v">${flooded.toFixed(0)}%</span>
     <span class="k">travel demand still connected</span><span class="v ${danger}">${access.toFixed(0)}%</span>
     <span class="k">city-wide efficiency</span><span class="v">${(sc.efficiency_ratio*100).toFixed(0)}%</span>`;
  if (sc.care_access != null) {
    const ca = sc.care_access * 100;
    html += `<span class="k">people with hospital access</span><span class="v ${ca<70?'danger':''}">${ca.toFixed(0)}%</span>`;
  }
  $("#floodStats").innerHTML = html;
}

function drawFlooded(ids) {
  const set = new Set(ids);
  if (floodLayer) map.removeLayer(floodLayer);
  floodLayer = L.layerGroup();
  ids.forEach((id) => {
    const nd = NODE_INDEX[id];
    if (nd) L.circleMarker([nd.lat, nd.lon], { radius: 5, color: "#1c6fd6",
      weight: 1, fillColor: "#3a98ff", fillOpacity: 0.85 }).addTo(floodLayer);
  });
  floodLayer.addTo(map);
}

function drawFloodChart(curve) {
  const svg = $("#floodChart"); const W = 280, H = 70, pad = 4;
  const n = curve.length;
  const x = (i) => pad + (W - 2*pad) * (i/(n-1));
  const y = (v) => pad + (H - 2*pad) * (1 - Math.max(0, Math.min(1, v)));
  let dLine = "", dArea = `M ${x(0)} ${y(0)} `;
  curve.forEach((p, i) => {
    dLine += (i?"L":"M") + ` ${x(i).toFixed(1)} ${y(p.access_retained).toFixed(1)} `;
    dArea += `L ${x(i).toFixed(1)} ${y(p.access_retained).toFixed(1)} `;
  });
  dArea += `L ${x(n-1)} ${y(0)} Z`;
  svg.innerHTML =
    `<defs><linearGradient id="fg" x1="0" y1="0" x2="0" y2="1">
       <stop offset="0%" stop-color="#3a98ff" stop-opacity="0.35"/>
       <stop offset="100%" stop-color="#3a98ff" stop-opacity="0.02"/></linearGradient></defs>
     <path d="${dArea}" fill="url(#fg)"/>
     <path d="${dLine}" fill="none" stroke="#3a98ff" stroke-width="2"/>`;
}

$("#reconBtn").onclick = async () => {
  const btn = $("#reconBtn");
  btn.disabled = true; btn.textContent = "Reconstructing…";
  toast("Skeletonising mask, building graph, healing gaps…", 8000);
  try {
    const center = (GRAPH && GRAPH.meta && GRAPH.meta.center) || null;
    const place = (GRAPH && GRAPH.meta && GRAPH.meta.name) || "";
    const r = await fetch("/api/load_mask", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo: true, center, place }) });
    const data = await r.json();
    if (data.error) { toast("⚠ " + data.error, 4000); return; }
    GRAPH = data; CLOSED = []; render();
    const rec = data.reconstruction;
    toast(`Healed ${rec.bridges_added} gaps · pieces ${rec.before.components}→${rec.after.components}`);
  } catch (e) { toast("Failed: " + e.message, 4000); }
  finally { btn.disabled = false; btn.textContent = "▶ Reconstruction demo (satellite mask → healed roads)"; }
};

if ($("#showHealed")) $("#showHealed").onchange = () => drawMap();

function readFileB64(file) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = rej;
    fr.readAsDataURL(file);
  });
}

$("#maskBtn").onclick = async () => {
  const f = $("#maskFile").files[0];
  if (!f) { toast("Choose a mask file first."); return; }
  const isGeo = /\.tiff?$/i.test(f.name);
  const bbox = [$("#bbS").value, $("#bbW").value, $("#bbN").value, $("#bbE").value].map(parseFloat);
  if (!isGeo && bbox.some(isNaN)) {
    toast("Plain image needs all four bbox values — or upload a GeoTIFF.", 4000); return;
  }
  const btn = $("#maskBtn"); btn.disabled = true; btn.textContent = "…";
  toast(isGeo ? "Reading GeoTIFF bounds & healing…" : "Healing your mask…", 8000);
  try {
    const mask_b64 = await readFileB64(f);
    const body = { mask_b64, name: f.name };
    if (!isGeo) body.bbox = bbox;
    const r = await fetch("/api/load_mask", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json();
    if (data.error) { toast("⚠ " + data.error, 5000); return; }
    GRAPH = data; CLOSED = []; render();
    const rec = data.reconstruction;
    toast(`Healed ${rec.bridges_added} gaps · pieces ${rec.before.components}→${rec.after.components}`);
  } catch (e) { toast("Failed: " + e.message, 5000); }
  finally { btn.disabled = false; btn.textContent = "Reconstruct from mask"; }
};

async function loadCity() {
  const place = $("#placeInput").value.trim();
  if (!place) { toast("Type a place name first."); return; }
  const radius = parseInt($("#radiusSel").value, 10);
  const btn = $("#loadBtn");
  btn.disabled = true; btn.textContent = "…";
  toast(`Fetching ${place} from OpenStreetMap…`, 8000);
  try {
    const demand = $("#demandChk") ? $("#demandChk").checked : true;
    const r = await fetch("/api/load_osm", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ place, radius_m: radius, demand }),
    });
    const data = await r.json();
    if (data.error) { toast("⚠ " + data.error, 5000); return; }
    GRAPH = data; CLOSED = [];
    render();
    toast(`Loaded ${data.n_nodes} junctions, ${data.n_edges} roads.`);
  } catch (e) {
    toast("Request failed: " + e.message, 5000);
  } finally {
    btn.disabled = false; btn.textContent = "Load";
  }
}

$("#loadBtn").onclick = loadCity;
$("#placeInput").addEventListener("keydown", (e) => { if (e.key === "Enter") loadCity(); });
$("#demoBtn").onclick = async () => {
  const r = await fetch("/api/load_demo", { method: "POST" });
  GRAPH = await r.json(); CLOSED = []; render();
  toast("Loaded the hand-made demo map.");
};

// ---- Glow-up B: live demand-flow particle field (betweenness, visualised) ----
let FLOW = null;
let CASCADE_FAILED = new Set();        // junctions failed by the running stress test

function flowColor(w, a) {
  // quiet -> busy -> gatekeeper corridor
  if (w < 0.45) { const t = w / 0.45; return `rgba(${110 + 122 * t},${130 + 80 * t},${150 - 40 * t},${a})`; }
  const t = Math.min(1, (w - 0.45) / 0.55);
  return `rgba(255,${210 - 120 * t},${74 - 24 * t},${a})`;   // amber -> red
}

function coreColor(w) {
  // hot, near-white core tinted by the corridor's load — keeps each particle distinct
  return w < 0.45 ? "rgba(228,246,255,0.96)" : "rgba(255,249,228,0.98)";
}

// build both the particles and a weighted pool of edges (for live re-routing)
function buildFlowData() {
  const feats = GRAPH.geojson.edges.features;
  let emax = 1e-9;
  feats.forEach((f) => { if (f.properties.criticality > emax) emax = f.properties.criticality; });
  const edgeList = [], parts = [];
  feats.forEach((f) => {
    const coords = f.geometry.coordinates;                // [[lon,lat],...]
    if (coords.length < 2) return;
    const w = f.properties.criticality / emax;            // 0..1 demand-weighted load
    const e = { u: f.properties.u, v: f.properties.v, coords, w };
    edgeList.push(e);
    const k = Math.round(w * w * 5);                      // critical corridors carry more
    for (let j = 0; j < k; j++)
      parts.push({ u: e.u, v: e.v, coords, w, p: Math.random(), speed: 0.0022 + 0.0056 * w });
  });
  // Keep canvas work light on large city graphs; busiest corridors are retained.
  if (parts.length > 500) parts.sort((a, b) => b.w - a.w).length = 500;
  // busiest-first, so a re-routed particle lands on a major surviving corridor
  edgeList.sort((a, b) => b.w - a.w);
  return { parts, edgeList };
}

// pick a living corridor to re-route a particle whose road just failed
function pickLiveEdge(dead) {
  const list = FLOW.edgeList, n = list.length;
  for (let tries = 0; tries < 30; tries++) {
    const e = list[(Math.random() * n) | 0];
    if (e.w > 0.15 && !dead.has(e.u) && !dead.has(e.v)) return e;
  }
  return null;
}

function pointAlong(coords, p) {
  const n = coords.length - 1, fp = p * n;
  let si = Math.floor(fp); let fr = fp - si;
  if (si >= n) { si = n - 1; fr = 1; }
  const a = coords[si], b = coords[si + 1];
  return [a[1] + (b[1] - a[1]) * fr, a[0] + (b[0] - a[0]) * fr];   // [lat,lon]
}

function flowFrame() {
  if (!FLOW || !FLOW.active) return;
  const ctx = FLOW.ctx, size = map.getSize();
  if (FLOW.canvas.width !== size.x || FLOW.canvas.height !== size.y) {
    FLOW.canvas.width = size.x; FLOW.canvas.height = size.y;
  }
  ctx.clearRect(0, 0, size.x, size.y);
  // every failed junction — clicked OR knocked out by the running stress test
  const dead = FLOW._dead || (FLOW._dead = new Set());
  dead.clear();
  CLOSED.forEach((id) => dead.add(String(id)));
  CASCADE_FAILED.forEach((id) => dead.add(id));

  // advance + project (store screen positions for the draw passes)
  for (const pt of FLOW.particles) {
    pt.p += pt.speed; if (pt.p >= 1) pt.p -= 1;
    if (dead.has(pt.u) || dead.has(pt.v)) {               // this road just failed...
      const ne = pickLiveEdge(dead);                      // ...re-route onto a survivor
      if (!ne) { pt._vis = false; continue; }             // nothing alive nearby: vanish
      pt.u = ne.u; pt.v = ne.v; pt.coords = ne.coords; pt.w = ne.w;
      pt.p = Math.random(); pt.speed = 0.0022 + 0.0056 * ne.w;
    }
    const cp = map.latLngToContainerPoint(pointAlong(pt.coords, pt.p));
    if (cp.x < -12 || cp.y < -12 || cp.x > size.x + 12 || cp.y > size.y + 12) { pt._vis = false; continue; }
    pt._cp = cp;
    pt._cpp = map.latLngToContainerPoint(pointAlong(pt.coords, Math.max(0, pt.p - pt.speed * 4)));
    pt._vis = true;
  }

  // pass 1 — radiant trails + soft halos (additive, gives the glow)
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  for (const pt of FLOW.particles) {
    if (!pt._vis) continue;
    ctx.strokeStyle = flowColor(pt.w, 0.42 + 0.30 * pt.w);
    ctx.lineWidth = 1.2 + 1.8 * pt.w;
    ctx.beginPath(); ctx.moveTo(pt._cpp.x, pt._cpp.y); ctx.lineTo(pt._cp.x, pt._cp.y); ctx.stroke();
    ctx.beginPath();
    ctx.fillStyle = flowColor(pt.w, 0.65);
    ctx.shadowColor = flowColor(pt.w, 1); ctx.shadowBlur = 6 + 7 * pt.w;
    ctx.arc(pt._cp.x, pt._cp.y, 2 + 1.8 * pt.w, 0, 6.2832); ctx.fill();
  }
  ctx.shadowBlur = 0;

  // pass 2 — crisp bright cores on top (each particle stays a distinct point)
  ctx.globalCompositeOperation = "source-over";
  for (const pt of FLOW.particles) {
    if (!pt._vis) continue;
    ctx.beginPath();
    ctx.fillStyle = coreColor(pt.w);
    ctx.arc(pt._cp.x, pt._cp.y, 1.1 + 1.0 * pt.w, 0, 6.2832); ctx.fill();
  }

  FLOW.raf = requestAnimationFrame(flowFrame);
}

function startFlow() {
  if (!GRAPH || !GRAPH.geojson) { toast("Load a city first."); return; }
  const cv = document.createElement("canvas");
  cv.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;z-index:450";
  const size = map.getSize(); cv.width = size.x; cv.height = size.y;
  $("#map").appendChild(cv);
  const data = buildFlowData();
  FLOW = { active: true, canvas: cv, ctx: cv.getContext("2d"),
           particles: data.parts, edgeList: data.edgeList, raf: null };
  FLOW.raf = requestAnimationFrame(flowFrame);
  $("#flowBtn").textContent = "■ Stop traffic flow";
  toast("Live flow on — brightness = how much cross-city traffic a road carries. Click a junction to fail it.");
}

function stopFlow() {
  CASCADE_FAILED = new Set();
  if (!FLOW) return;
  FLOW.active = false;
  if (FLOW.raf) cancelAnimationFrame(FLOW.raf);
  if (FLOW.canvas && FLOW.canvas.parentNode) FLOW.canvas.parentNode.removeChild(FLOW.canvas);
  FLOW = null;
  const b = $("#flowBtn"); if (b) b.textContent = "▶ Show live traffic flow";
}

$("#flowBtn").onclick = () => { if (FLOW && FLOW.active) stopFlow(); else startFlow(); };

// ---- Glow-up C: confidence-aware healing replay (turns healed gaps legible) ----
const HEAL_MAX_GAP = 130;                                  // matches the server's heal_kw
const healConfidence = (len) => Math.max(0.3, Math.min(0.97, 1 - len / (HEAL_MAX_GAP * 1.15)));

function renderHealConf(rows, rec) {
  const box = $("#healConf");
  box.classList.remove("hidden");
  const lo = rows.filter((r) => r.cf < 0.6).length;
  const avg = rows.reduce((s, r) => s + r.cf, 0) / Math.max(1, rows.length);
  box.innerHTML =
    `<div class="hc-rigor">connectivity <b>without</b> healed (raw): ${rec.before.components} pieces ·
       ${(rec.before.lcc_frac * 100).toFixed(0)}%<br>
       <b>with</b> healed: ${rec.after.components} pieces · ${(rec.after.lcc_frac * 100).toFixed(0)}%</div>
     <div class="hc-line">${rows.length} bridges invented · avg confidence
       <b>${(avg * 100).toFixed(0)}%</b> ·
       <span class="${lo ? "hc-warn" : ""}">${lo} low-confidence (&lt;60%)</span></div>
     <div class="hc-note">confidence ∝ shorter gap. Low-confidence links are candidates,
       not certainties — resilience should be read with this caveat.</div>`;
}

async function replayHealing() {
  const rec = GRAPH.reconstruction;
  if (!rec) { toast("Reconstruct or upload a mask first."); return; }
  const healed = GRAPH.geojson.edges.features.filter((f) => f.properties.healed);
  if (!healed.length) { toast("No healed links to replay."); return; }
  stopFlow();
  const btn = $("#healReplayBtn"); btn.disabled = true; btn.textContent = "Replaying…";
  [edgeLayer, nodeLayer, rerouteLayer, floodLayer, facilityLayer, repairLayer]
    .forEach((l) => l && map.removeLayer(l));

  // beat 1 — the raw, broken extraction
  const rawLayer = L.geoJSON(
    { type: "FeatureCollection", features: GRAPH.geojson.edges.features.filter((f) => !f.properties.healed) },
    { style: { color: "#5a6473", weight: 1.6, opacity: 0.5 } }).addTo(map);
  try { map.fitBounds(rawLayer.getBounds(), { padding: [40, 40] }); } catch (e) {}
  toast(`Raw extraction: ${rec.before.components} pieces, largest area ${(rec.before.lcc_frac * 100).toFixed(0)}%`, 2600);
  await sleep(1200);

  // beat 2 — flash where the gaps are
  const gapLayer = L.layerGroup().addTo(map);
  healed.forEach((f) => {
    const c = f.geometry.coordinates; const m = c[Math.floor(c.length / 2)];
    L.circleMarker([m[1], m[0]], { radius: 5, color: "#ff5a5a", weight: 2,
      fillColor: "#ff5a5a", fillOpacity: 0.5 }).addTo(gapLayer);
  });
  await sleep(900);
  map.removeLayer(gapLayer);

  // beat 3 — snap the healed bridges in, weakest links most transparent
  const healLayer = L.layerGroup().addTo(map);
  const rows = [];
  for (let i = 0; i < healed.length; i++) {
    const f = healed[i];
    const latlngs = f.geometry.coordinates.map((c) => [c[1], c[0]]);
    const cf = healConfidence(f.properties.length || 0);
    L.polyline(latlngs, { color: "#5ad19a", weight: 2 + 2.5 * cf,
      opacity: 0.4 + 0.55 * cf, dashArray: "5 5" }).addTo(healLayer);
    rows.push({ cf });
    if (i % 3 === 0 || i === healed.length - 1) renderHealConf(rows, rec);
    await sleep(140);
  }
  renderHealConf(rows, rec);
  btn.disabled = false; btn.textContent = "▶ Replay healing (with confidence)";
  await sleep(2600);
  map.removeLayer(rawLayer); map.removeLayer(healLayer);
  drawMap();
}

$("#healReplayBtn").onclick = replayHealing;

loadGraph();
