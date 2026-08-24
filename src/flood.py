"""
Titan Arch  --  flood.py  (Layer 2: hazard scenarios)

A flood doesn't knock out random junctions -- it drowns the low ground first,
so failures are spatially correlated. Given each junction's elevation, we raise
a water line and fail every junction beneath it together, then measure what the
city has left: how much travel demand is still connected to the main network.

This turns the "what-if" from clicking random nodes into a realistic disaster
with a dial: how deep a flood can the city survive, and who gets cut off first.
"""

from __future__ import annotations

import networkx as nx

from roadgraph import _flow_sum, _pick_sources


def elevation_range(elev):
    vals = [v for v in elev.values()]
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def _demand_in_largest(G, weight):
    """Total demand sitting in the largest connected piece (a proxy for how much
    of the city is still usefully reachable)."""
    if G.number_of_nodes() == 0:
        return 0.0
    biggest = max(nx.connected_components(G), key=len)
    if weight:
        return sum(weight.get(n, 0.0) for n in biggest)
    return float(len(biggest))


def flood_scenario(G, elev, depth_m, weight=None, sources=None, base_flow=None,
                   base_demand=None, care_nodes=None, reach_m=2500):
    """
    Raise water to (lowest elevation + depth_m); fail every junction at or below
    it. Returns the flooded junctions and the damage.

    access_retained : share of total travel demand still connected to the main
                      road network (1.0 = unaffected, 0 = city severed).
    efficiency_ratio: demand-weighted routing efficiency vs the dry network.
    """
    lo, _hi = elevation_range(elev)
    water = lo + depth_m

    flooded = [n for n in G if n in elev and elev[n] <= water]
    H = G.copy()
    H.remove_nodes_from(flooded)

    total_demand = base_demand if base_demand is not None else (
        sum(weight.values()) if weight else G.number_of_nodes())
    access = _demand_in_largest(H, weight)
    access_retained = access / total_demand if total_demand else 0.0

    if sources is None:
        sources = _pick_sources(G, min(200, G.number_of_nodes()))
    if base_flow is None:
        base_flow = _flow_sum(G, sources, weight) or 1e-12
    eff_ratio = _flow_sum(H, sources, weight) / base_flow

    # hospital access on the flooded network: facilities that drowned are gone
    care_access = None
    if care_nodes:
        import emergency as em
        live_care = [c for c in care_nodes if c in H]
        if live_care:
            dist = em.access_distances(H, live_care)
            care_access = em.served_fraction(H, live_care, weight, reach_m=reach_m,
                                              dist=dist)
        else:
            care_access = 0.0

    return {
        "depth_m": depth_m,
        "water_level_m": water,
        "flooded": [str(n) for n in flooded],
        "n_flooded": len(flooded),
        "frac_junctions_flooded": len(flooded) / G.number_of_nodes() if G.number_of_nodes() else 0,
        "access_retained": max(0.0, min(1.0, access_retained)),
        "efficiency_ratio": max(0.0, min(1.0, eff_ratio)),
        "lcc_frac": (max((len(c) for c in nx.connected_components(H)), default=0)
                     / H.number_of_nodes()) if H.number_of_nodes() else 0.0,
        "care_access": care_access,
    }


def flood_curve(G, elev, weight=None, steps=12, max_depth=None):
    """access_retained at a ladder of water depths -- the city's flood tolerance."""
    lo, hi = elevation_range(elev)
    if max_depth is None:
        max_depth = max(1.0, hi - lo)
    sources = _pick_sources(G, min(200, G.number_of_nodes()))
    base_flow = _flow_sum(G, sources, weight) or 1e-12
    base_demand = sum(weight.values()) if weight else G.number_of_nodes()
    curve = []
    for i in range(steps + 1):
        depth = max_depth * i / steps
        sc = flood_scenario(G, elev, depth, weight, sources, base_flow, base_demand)
        curve.append({"depth_m": round(depth, 1),
                      "access_retained": sc["access_retained"],
                      "frac_flooded": sc["frac_junctions_flooded"]})
    return {"min_elev": lo, "max_elev": hi, "max_depth": max_depth, "curve": curve}
