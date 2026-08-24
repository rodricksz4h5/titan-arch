"""
Titan Arch  --  capacity.py  (Layer 4: capacity & congestion)

A failed junction doesn't make traffic vanish -- it dumps that traffic onto
other roads, which clog. This layer models that with standard transportation
engineering:

  * every road gets a CAPACITY from its class (a motorway carries far more than
    a residential street),
  * travel demand (Layer 1) is routed onto shortest paths -> each road gets a
    traffic VOLUME (a demand-weighted traffic assignment),
  * the volume/capacity (V/C) ratio gives congestion, and the BPR function
    turns it into a realistic travel-time penalty.

When a junction fails, we re-route and recompute -- and watch V/C spike on the
roads that have to absorb the displaced traffic.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

# typical one-direction capacity (veh/hour) and free-flow speed (km/h) by class
CAPACITY = {"motorway": 4000, "trunk": 3000, "primary": 1800, "secondary": 1200,
            "tertiary": 800, "unclassified": 600, "residential": 600,
            "living_street": 300, "road": 600}
SPEED_KMH = {"motorway": 80, "trunk": 60, "primary": 50, "secondary": 40,
             "tertiary": 35, "unclassified": 30, "residential": 30,
             "living_street": 20, "road": 30}
ALPHA, BETA = 0.15, 4.0           # BPR parameters (Bureau of Public Roads)


def _cap(G, u, v):
    return CAPACITY.get(G[u][v].get("highway", "residential"), 600)


def _t0_min(G, u, v):
    hw = G[u][v].get("highway", "residential")
    spd = SPEED_KMH.get(hw, 30)
    return (G[u][v]["length"] / 1000.0) / spd * 60.0      # minutes


def assign_traffic(G, weight=None, sample=120, seed=2):
    """
    Demand-weighted all-or-nothing assignment: route each origin's demand to
    every destination along the shortest path, accumulating volume on edges.
    Returns {edge_key: volume} where edge_key = "u|v" (sorted).
    """
    nodes = list(G)
    n = len(nodes)
    w = {v: (weight.get(v, 1.0) if weight else 1.0) for v in nodes}

    if sample and sample < n:
        import random
        sources = random.Random(seed).sample(nodes, sample)
        scale = n / sample
    else:
        sources = nodes
        scale = 1.0

    vol = defaultdict(float)
    for s in sources:
        _, paths = nx.single_source_dijkstra(G, s, weight="length")
        ws = w[s]
        for t, path in paths.items():
            if t == s or len(path) < 2:
                continue
            f = ws * w[t]
            for a, b in zip(path, path[1:]):
                vol[f"{a}|{b}" if str(a) < str(b) else f"{b}|{a}"] += f
    for k in vol:
        vol[k] *= scale
    return vol


def _calibrate(G, vol, target_p85=0.85):
    """Scale arbitrary demand-volume units so the 85th-percentile road runs near
    capacity (a realistic baseline). Returns the scale factor."""
    ratios = []
    for u, v in G.edges():
        key = f"{u}|{v}" if str(u) < str(v) else f"{v}|{u}"
        ratios.append(vol.get(key, 0.0) / _cap(G, u, v))
    ratios.sort()
    if not ratios:
        return 1.0
    p85 = ratios[int(0.85 * (len(ratios) - 1))]
    return (target_p85 / p85) if p85 > 0 else 1.0


def congestion(G, weight=None, sample=120, scale=None):
    """
    Baseline congestion: V/C per road + system stats. Returns a dict with
    per-edge vc (for colouring) and a summary.
    """
    vol = assign_traffic(G, weight, sample)
    if scale is None:
        scale = _calibrate(G, vol)

    vc, total_vehmin, total_extra, overloaded_len, total_len = {}, 0.0, 0.0, 0.0, 0.0
    worst = []
    for u, v in G.edges():
        key = f"{u}|{v}" if str(u) < str(v) else f"{v}|{u}"
        v_over_c = vol.get(key, 0.0) * scale / _cap(G, u, v)
        vc[key] = v_over_c
        t0 = _t0_min(G, u, v)
        t = t0 * (1 + ALPHA * (v_over_c ** BETA))
        length = G[u][v]["length"]
        total_len += length
        total_vehmin += v_over_c * t
        total_extra += v_over_c * (t - t0)
        if v_over_c > 1.0:
            overloaded_len += length
        worst.append((v_over_c, key, G[u][v].get("highway", "")))
    worst.sort(reverse=True)

    return {
        "vc": vc,
        "scale": scale,
        "pct_overloaded": (overloaded_len / total_len * 100) if total_len else 0.0,
        "total_delay_index": total_extra,
        "worst": [{"edge": k, "vc": round(r, 2), "class": c} for r, k, c in worst[:8]],
    }


def _reroute_delay(G, H, weight, sample, seed=2):
    """Extra travel time the surviving trips must pay after the failure. Measured
    over the SAME origin-destination pairs (only those still connected), so it is
    apples-to-apples and never negative."""
    nodes = list(G)
    w = {v: (weight.get(v, 1.0) if weight else 1.0) for v in nodes}
    import random
    srcs = random.Random(seed).sample(nodes, min(sample, len(nodes)))
    base_t = new_t = 0.0
    for s in srcs:
        if s not in H:
            continue
        dg = nx.single_source_dijkstra_path_length(G, s, weight="length")
        dh = nx.single_source_dijkstra_path_length(H, s, weight="length")
        for t, dgt in dg.items():
            if t == s or t not in dh:        # only pairs still connected in H
                continue
            f = w[s] * w[t]
            base_t += f * dgt
            new_t += f * dh[t]
    return ((new_t - base_t) / base_t * 100) if base_t else 0.0


def congestion_under_failure(G, dead_nodes, weight=None, sample=120, scale=None):
    """Recompute congestion after junctions fail; report the extra delay and the
    roads whose V/C jumps the most as they absorb the displaced traffic."""
    base = congestion(G, weight, sample, scale)
    H = G.copy()
    H.remove_nodes_from(dead_nodes)
    pert = congestion(H, weight, sample, scale=base["scale"])

    jumps = []
    for key, r in pert["vc"].items():
        b = base["vc"].get(key, 0.0)
        if r - b > 0.05:
            jumps.append({"edge": key, "from": round(b, 2), "to": round(r, 2)})
    jumps.sort(key=lambda j: j["to"] - j["from"], reverse=True)
    return {
        "base_overloaded": base["pct_overloaded"],
        "new_overloaded": pert["pct_overloaded"],
        "extra_delay_pct": _reroute_delay(G, H, weight, sample),
        "spiking_roads": jumps[:8],
        "vc": pert["vc"],
    }
