"""
Titan Arch  --  roadgraph.py  (the analysis brain)

We use networkx for the well-solved graph math (shortest paths, betweenness,
efficiency, components) and keep our own code for the parts that are the
actual product:  the resilience stress-test orchestration + the Resilience
Index, and the interactive node-ablation used by the click-to-fail dashboard.

A road network here is a networkx.Graph where:
  - each node has attributes  lat, lon
  - each edge has attribute    length  (metres) used as the travel weight
"""

from __future__ import annotations

import math
import networkx as nx


# --------------------------------------------------------------------------- #
#  geometry helper
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def add_edge(G, u, v, length=None):
    """Add an undirected road segment, auto-computing length from coords."""
    if u == v:
        return
    if length is None:
        length = haversine_m(G.nodes[u]["lat"], G.nodes[u]["lon"],
                             G.nodes[v]["lat"], G.nodes[v]["lon"])
    G.add_edge(u, v, length=float(length))


# --------------------------------------------------------------------------- #
#  criticality  (who is a gatekeeper)
# --------------------------------------------------------------------------- #
def criticality(G, sample=None, weight=None):
    """
    Betweenness centrality on every node: the share of shortest trips that pass
    through it. High = chokepoint.

    If `weight` (node_id -> demand) is given, trips are weighted by real demand
    (a trip s->t counts as demand[s]*demand[t]) instead of every junction pair
    counting equally -- Layer 1. networkx has no endpoint-weighted betweenness,
    so we run a weighted Brandes ourselves. Otherwise we use networkx.
    """
    n = G.number_of_nodes()
    if n < 3:
        return {v: 0.0 for v in G}
    if weight is not None:
        return demand_weighted_betweenness(G, weight, sample=sample)
    k = min(sample, n) if sample else None
    return nx.betweenness_centrality(G, k=k, weight="length",
                                     normalized=True, seed=1)


def demand_weighted_betweenness(G, weight, sample=None, seed=1):
    """
    Weighted Brandes: a trip from s to t contributes weight[s]*weight[t] to the
    junctions on its shortest path. Sampled over sources for speed (unbiased,
    scaled back up). Returns a dict normalised to [0,1] by its own max.
    """
    import heapq
    import random

    nodes = list(G)
    n = len(nodes)
    w = {v: max(float(weight.get(v, 0.0)), 0.0) for v in nodes}

    if sample and sample < n:
        sources = random.Random(seed).sample(nodes, sample)
        scale = n / sample
    else:
        sources = nodes
        scale = 1.0

    cb = dict.fromkeys(nodes, 0.0)
    for s in sources:
        S, P = [], {v: [] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0); sigma[s] = 1.0
        dist = {s: 0.0}; seen = {s: 0.0}
        pq = [(0.0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist.get(v, math.inf):
                continue
            S.append(v)
            for u, ed in G[v].items():
                nd = d + ed["length"]
                if u not in seen or nd < seen[u]:
                    seen[u] = nd; dist[u] = nd
                    sigma[u] = sigma[v]; P[u] = [v]
                    heapq.heappush(pq, (nd, u))
                elif abs(nd - seen[u]) < 1e-9:
                    sigma[u] += sigma[v]; P[u].append(v)
        # accumulate dependency, with each target weighted by its demand
        delta = dict.fromkeys(nodes, 0.0)
        while S:
            u = S.pop()
            coeff = (w[u] + delta[u]) / sigma[u] if sigma[u] else 0.0
            for v in P[u]:
                delta[v] += sigma[v] * coeff
            if u != s:
                cb[u] += w[s] * delta[u]

    for v in cb:                       # undirected: each pair seen twice
        cb[v] *= scale / 2.0
    mx = max(cb.values()) if cb else 0.0
    if mx > 0:
        for v in cb:
            cb[v] /= mx                # normalise to [0,1] for display/colour
    return cb


# --------------------------------------------------------------------------- #
#  global efficiency  (how easily the whole city gets around)
# --------------------------------------------------------------------------- #
def global_efficiency(G, sample=None, seed=0, weight=None):
    """
    Mean of 1/distance over node pairs. Disconnected pairs count as 0, so when
    the network fragments this drops -- the heart of the resilience measure.
    Sampled over random sources for large graphs.

    If `weight` (node demand) is given, each pair (s,t) is weighted by
    demand[s]*demand[t], so fragmenting a busy area hurts the score far more
    than fragmenting an empty one (Layer 1: demand-aware resilience).
    """
    nodes = list(G)
    n = len(nodes)
    if n < 2:
        return 0.0

    if sample and sample < n:
        import random
        sources = random.Random(seed).sample(nodes, sample)
        src_scale = n / sample
    else:
        sources = nodes
        src_scale = 1.0

    if weight is None:
        total = 0.0
        for s in sources:
            for t, d in nx.single_source_dijkstra_path_length(G, s, weight="length").items():
                if t != s and d > 0:
                    total += 1.0 / d
        return total * src_scale / (n * (n - 1)) if n > 1 else 0.0

    w = {v: max(float(weight.get(v, 0.0)), 0.0) for v in nodes}
    num = den = 0.0
    for s in sources:
        ws = w[s]
        for t, d in nx.single_source_dijkstra_path_length(G, s, weight="length").items():
            if t != s and d > 0:
                num += ws * w[t] / d
    # normalise by total demanded pair-weight (so it stays a 0..~ comparable scale)
    tot_w = sum(w.values())
    den = (tot_w * tot_w - sum(x * x for x in w.values()))
    return (num * src_scale) / den if den else 0.0


def largest_component_fraction(G):
    if G.number_of_nodes() == 0:
        return 0.0
    return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()


def average_path_length(G, sources):
    """Mean road distance over all reachable junction pairs from a fixed probe
    set. This is the quantity the spec's Resilience Index is built on."""
    total, count = 0.0, 0
    for s in sources:
        if s not in G:
            continue
        for t, d in nx.single_source_dijkstra_path_length(G, s, weight="length").items():
            if t != s and d > 0:
                total += d
                count += 1
    return total / count if count else float("inf")


def _pick_sources(G, sample, seed=0):
    """A FIXED probe set of source junctions, sampled once from the intact
    network. Reusing the same sources before/after a failure makes the
    efficiency ratio apples-to-apples (and guaranteed <= 1)."""
    nodes = list(G)
    if sample and sample < len(nodes):
        import random
        return random.Random(seed).sample(nodes, sample)
    return nodes


def _flow_sum(G, sources, weight=None):
    """Raw reachability score over a fixed source set: sum of (demand-weighted)
    1/distance to every reachable junction. Removing junctions can only drop
    terms, never add them -- so this never increases when the network degrades.
    Sources/targets absent from G simply contribute nothing."""
    total = 0.0
    for s in sources:
        if s not in G:
            continue
        ws = weight[s] if weight else 1.0
        for t, d in nx.single_source_dijkstra_path_length(G, s, weight="length").items():
            if t != s and d > 0:
                total += ws * (weight[t] if weight else 1.0) / d
    return total


# --------------------------------------------------------------------------- #
#  resilience stress-test  (our metric: pull the worst nodes, watch it break)
# --------------------------------------------------------------------------- #
def resilience_stress_test(G, top_k=10, sample=None, weight=None):
    """
    Remove the highest-criticality nodes one at a time; record how global
    efficiency and the largest connected piece collapse.

    Resilience Index = efficiency after removing the top_k gatekeepers,
    relative to the intact network (1.0 = untouched, ~0 = devastated).
    A low index means a few junctions hold the whole city together.

    With `weight`, both the ranking and the efficiency are demand-weighted.
    """
    bc = criticality(G, sample=sample, weight=weight)
    ranked = sorted(bc, key=bc.get, reverse=True)

    # don't wipe tiny networks: cap removals at ~a third of the nodes
    n = G.number_of_nodes()
    eff_k = max(1, min(top_k, n // 3))

    # fixed probe set -> before/after are comparable, ratio guaranteed <= 1
    sources = _pick_sources(G, sample)
    base_flow = _flow_sum(G, sources, weight) or 1e-12
    base_eff = global_efficiency(G, sample=sample, weight=weight)
    steps, removed = [], []
    for nid in ranked[:eff_k]:
        removed.append(nid)
        H = G.copy()
        H.remove_nodes_from(removed)
        eff = _flow_sum(H, sources, weight)
        steps.append({
            "removed": nid,
            "efficiency": eff,
            "eff_ratio": eff / base_flow,
            "lcc_frac": largest_component_fraction(H),
        })

    # spec-defined Resilience Index = baseline APL / perturbed APL (over the
    # junctions still reachable after removing the top gatekeepers)
    base_apl = average_path_length(G, sources)
    if removed:
        Hf = G.copy()
        Hf.remove_nodes_from(removed)
        pert_apl = average_path_length(Hf, sources)
    else:
        pert_apl = base_apl
    ri_apl = (base_apl / pert_apl) if (pert_apl and pert_apl != float("inf")) else 0.0

    return {
        "base_efficiency": base_eff,
        "steps": steps,
        "resilience_index": steps[-1]["eff_ratio"] if steps else 1.0,
        "resilience_index_apl": ri_apl,         # per the spec's definition
        "base_apl": base_apl,
        "ranked_nodes": ranked,
        "betweenness": bc,
    }


# --------------------------------------------------------------------------- #
#  interactive ablation  (powers the click-to-fail demo)
# --------------------------------------------------------------------------- #
def ablate(G, dead_nodes, probe_pairs=None, sample=None, weight=None):
    """
    Simulate closing one or more junctions. Returns the efficiency hit and,
    optionally, how specific origin->destination trips get longer (or cut off).

    probe_pairs : list of (src, dst) node ids to report detailed reroute cost.
    """
    dead = set(dead_nodes)
    H = G.copy()
    H.remove_nodes_from(dead)

    sources = _pick_sources(G, sample)
    base_eff = _flow_sum(G, sources, weight) or 1e-12
    new_eff = _flow_sum(H, sources, weight)

    trips = []
    for s, t in (probe_pairs or []):
        if s in dead or t in dead:
            trips.append({"src": s, "dst": t, "status": "endpoint_closed"})
            continue
        try:
            base_d = nx.shortest_path_length(G, s, t, weight="length")
        except nx.NetworkXNoPath:
            base_d = None
        try:
            new_d = nx.shortest_path_length(H, s, t, weight="length")
            new_path = nx.shortest_path(H, s, t, weight="length")
        except nx.NetworkXNoPath:
            new_d, new_path = None, None
        # convert distance -> travel time at an assumed urban road speed
        kmh = 30.0
        to_min = lambda m: (m / 1000.0) / kmh * 60.0 if m is not None else None
        trips.append({
            "src": s, "dst": t,
            "base_m": base_d, "new_m": new_d,
            "delta_m": (new_d - base_d) if (base_d and new_d) else None,
            "base_min": to_min(base_d), "new_min": to_min(new_d),
            "delta_min": (to_min(new_d) - to_min(base_d)) if (base_d and new_d) else None,
            "status": "cut_off" if new_d is None else "rerouted",
            "new_path": new_path,
        })

    return {
        "closed": list(dead),
        "base_efficiency": base_eff,
        "new_efficiency": new_eff,
        "efficiency_ratio": new_eff / base_eff,
        "lcc_frac": largest_component_fraction(H),
        "trips": trips,
    }


# --------------------------------------------------------------------------- #
#  repair recommender  (our twist: which ONE new link helps the city most?)
# --------------------------------------------------------------------------- #
def _node_dist_m(G, u, v):
    return haversine_m(G.nodes[u]["lat"], G.nodes[u]["lon"],
                       G.nodes[v]["lat"], G.nodes[v]["lon"])


def repair_candidates(G, max_dist_m=350, min_detour=3.0, limit=40):
    """
    Propose new road links worth considering: pairs of junctions that are
    physically CLOSE (could plausibly be joined) but are FAR apart by road
    (a big detour) or outright disconnected. Those are the gaps that, if
    bridged, should help the most.

    Returns a list of (u, v, straight_m, detour_ratio) sorted best-first.
    """
    # spatial buckets so we only test physically-near pairs (not all n^2)
    nodes = list(G)
    if len(nodes) < 4:
        return []
    lat0 = sum(G.nodes[n]["lat"] for n in nodes) / len(nodes)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.2, math.cos(math.radians(lat0)))
    cell = max_dist_m

    buckets = {}
    for n in nodes:
        gx = int((G.nodes[n]["lon"] * m_per_deg_lon) // cell)
        gy = int((G.nodes[n]["lat"] * m_per_deg_lat) // cell)
        buckets.setdefault((gx, gy), []).append(n)

    seen, cands = set(), []
    for (gx, gy), members in buckets.items():
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near += buckets.get((gx + dx, gy + dy), [])
        for u in members:
            for v in near:
                if u == v:
                    continue
                key = (u, v) if str(u) < str(v) else (v, u)
                if key in seen or G.has_edge(u, v):
                    continue
                seen.add(key)
                d = _node_dist_m(G, u, v)
                if d > max_dist_m or d < 1.0:
                    continue
                try:
                    road = nx.shortest_path_length(G, u, v, weight="length")
                    detour = road / d
                except nx.NetworkXNoPath:
                    detour = float("inf")          # disconnected -> top priority
                if detour >= min_detour:
                    cands.append((u, v, d, detour))

    cands.sort(key=lambda c: c[3], reverse=True)
    return cands[:limit]


def recommend_repairs(G, top_n=3, sample=None, max_eval=24,
                      screen_sample=120, ri_before=None, weight=None):
    """
    Evaluate the candidate links by how much they raise global efficiency, and
    return the best few. For the single best we also report the Resilience
    Index before vs after, so the payoff is concrete.

    screen_sample : a lighter sample used to rank the many candidates fast.
    ri_before     : pass the already-computed Resilience Index to skip redoing it.
    weight        : node demand, so repairs are judged by who actually benefits.
    """
    scr = min(screen_sample, sample) if sample else screen_sample
    base_eff = global_efficiency(G, sample=scr, weight=weight) or 1e-12
    cands = repair_candidates(G)[:max_eval]
    scored = []
    for u, v, d, detour in cands:
        H = G.copy()
        H.add_edge(u, v, length=d)
        eff = global_efficiency(H, sample=scr, weight=weight)
        scored.append({
            "u": str(u), "v": str(v),
            "u_lat": G.nodes[u]["lat"], "u_lon": G.nodes[u]["lon"],
            "v_lat": G.nodes[v]["lat"], "v_lon": G.nodes[v]["lon"],
            "length_m": round(d, 1),
            "detour_ratio": None if detour == float("inf") else round(detour, 1),
            "reconnects": detour == float("inf"),
            "eff_gain_pct": round(100 * (eff - base_eff) / base_eff, 2),
        })
    scored.sort(key=lambda s: s["eff_gain_pct"], reverse=True)
    best = scored[:top_n]

    # concrete payoff for the top pick: resilience index before vs after
    if ri_before is None:
        ri_before = resilience_stress_test(G, sample=sample, weight=weight)["resilience_index"]
    ri_after = None
    if best:
        b = best[0]
        H = G.copy()
        H.add_edge(_match(G, b["u"]), _match(G, b["v"]), length=b["length_m"])
        ri_after = resilience_stress_test(H, sample=sample, weight=weight)["resilience_index"]

    return {"base_efficiency": base_eff, "suggestions": best,
            "resilience_before": ri_before, "resilience_after": ri_after}


def _match(G, sid):
    """Find the node whose str() matches sid (ids may be int or str)."""
    if sid in G:
        return sid
    for n in G:
        if str(n) == str(sid):
            return n
    return sid


# --------------------------------------------------------------------------- #
#  geojson export for the map dashboard
# --------------------------------------------------------------------------- #
def to_geojson(G, betweenness=None):
    """Edges as LineStrings (coloured by max endpoint criticality) + node points."""
    bc = betweenness or {}
    feats = []
    for u, v, data in G.edges(data=True):
        crit = max(bc.get(u, 0.0), bc.get(v, 0.0))
        # use the real road polyline if we kept it (from OSM), else a straight chord
        coords = data.get("geometry") or [
            [G.nodes[u]["lon"], G.nodes[u]["lat"]],
            [G.nodes[v]["lon"], G.nodes[v]["lat"]],
        ]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"u": str(u), "v": str(v),
                           "length": round(data.get("length", 0), 1),
                           "criticality": crit,
                           "healed": bool(data.get("healed", False))},
        })
    nodes = []
    for nid, d in G.nodes(data=True):
        nodes.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
            "properties": {"id": str(nid), "criticality": bc.get(nid, 0.0)},
        })
    return {"edges": {"type": "FeatureCollection", "features": feats},
            "nodes": {"type": "FeatureCollection", "features": nodes}}


# --------------------------------------------------------------------------- #
#  tiny hand-made map for Step 0 (two clusters joined by one bridge)
# --------------------------------------------------------------------------- #
def demo_graph():
    G = nx.Graph()
    pts = {
        0: (12.9700, 77.5900), 1: (12.9700, 77.5912),
        2: (12.9690, 77.5900), 3: (12.9690, 77.5912),
        4: (12.9690, 77.5930),                          # the bridge
        5: (12.9700, 77.5948), 6: (12.9700, 77.5960),
        7: (12.9690, 77.5948), 8: (12.9690, 77.5960),
    }
    for nid, (la, lo) in pts.items():
        G.add_node(nid, lat=la, lon=lo)
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2),
                 (5, 6), (5, 7), (6, 8), (7, 8), (5, 8), (6, 7),
                 (3, 4), (4, 7)]:
        add_edge(G, u, v)
    return G


if __name__ == "__main__":
    G = demo_graph()
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    res = resilience_stress_test(G, top_k=3)
    bc = res["betweenness"]
    print("top gatekeepers:", sorted(bc, key=bc.get, reverse=True)[:3])
    print(f"base efficiency = {res['base_efficiency']:.6f}")
    for s in res["steps"]:
        print(f"  remove {s['removed']}: eff_ratio={s['eff_ratio']:.3f} "
              f"largest_piece={s['lcc_frac']:.2f}")
    print(f"resilience index (after 3) = {res['resilience_index']:.3f}")
    ab = ablate(G, [4], probe_pairs=[(0, 8)])
    print("close bridge node 4 -> trip 0->8:", ab["trips"][0]["status"])
