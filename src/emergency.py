"""
Titan Arch  --  emergency.py  (Layer 3: access to emergency care)

Resilience isn't only "can people move" -- it's "can people reach a hospital".
This layer pulls hospitals, clinics and fire stations from OpenStreetMap, maps
them onto the road network, and answers:

  * what share of the population can reach emergency care (within reach)?
  * which junctions is that access most dependent on -- i.e. whose failure (or
    flooding) would strand the most people from care?

These "emergency gatekeepers" are often different from traffic gatekeepers, and
they're the ones that matter in a disaster.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import requests
from scipy.spatial import cKDTree

from osm import OVERPASS, UA

FACILITIES = '["amenity"~"hospital|clinic|fire_station"]'
DEFAULT_REACH_M = 2500          # ~ what's reachable quickly by road


def _to_xy(lat, lon, lat0):
    return (lon * 111_320.0 * math.cos(math.radians(lat0)), lat * 111_320.0)


def fetch_facilities(bbox, retries=3):
    s, w, n, e = bbox
    q = f"""
    [out:json][timeout:60];
    ( node{FACILITIES}({s},{w},{n},{e});
      way{FACILITIES}({s},{w},{n},{e});
      relation{FACILITIES}({s},{w},{n},{e}); );
    out center;
    """
    import time
    last = None
    for i in range(retries):
        try:
            r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=90)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 + 3 * i)
    else:
        raise RuntimeError(f"facility fetch failed: {last}")

    out = []
    for el in js["elements"]:
        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif "lat" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            continue
        out.append({"lat": lat, "lon": lon,
                    "kind": el.get("tags", {}).get("amenity", "facility"),
                    "name": el.get("tags", {}).get("name", "")})
    return out


def map_to_junctions(G, facilities):
    """Snap each facility to its nearest junction; return the set of care nodes."""
    nodes = list(G)
    if not nodes or not facilities:
        return []
    lat0 = sum(G.nodes[v]["lat"] for v in nodes) / len(nodes)
    node_xy = np.array([_to_xy(G.nodes[v]["lat"], G.nodes[v]["lon"], lat0) for v in nodes])
    tree = cKDTree(node_xy)
    fxy = np.array([_to_xy(f["lat"], f["lon"], lat0) for f in facilities])
    _, idx = tree.query(fxy, k=1)
    return list({nodes[i] for i in np.atleast_1d(idx)})


def access_distances(G, care_nodes):
    """Road distance from every junction to the nearest care facility."""
    if not care_nodes:
        return {}
    return nx.multi_source_dijkstra_path_length(G, set(care_nodes), weight="length")


def served_fraction(G, care_nodes, weight=None, reach_m=DEFAULT_REACH_M, dist=None):
    """Share of travel demand within reach_m (by road) of any facility."""
    if dist is None:
        dist = access_distances(G, care_nodes)
    total = sum(weight.values()) if weight else G.number_of_nodes()
    if not total:
        return 0.0
    served = 0.0
    for n in G:
        if dist.get(n, math.inf) <= reach_m:
            served += (weight.get(n, 0.0) if weight else 1.0)
    return served / total


def access_criticality(G, care_nodes, weight=None):
    """
    Emergency-dependency of each junction: the travel demand whose shortest path
    to its nearest facility passes THROUGH it. High = a lot of people rely on
    this junction to reach care; lose it and they're cut off or detoured.
    """
    cb = dict.fromkeys(G, 0.0)
    if not care_nodes:
        return cb
    _, paths = nx.multi_source_dijkstra(G, set(care_nodes), weight="length")
    for v, path in paths.items():
        wv = weight.get(v, 0.0) if weight else 1.0
        if wv <= 0 or len(path) < 2:
            continue
        for u in path[1:-1]:           # intermediate junctions carry the reliance
            cb[u] += wv
    mx = max(cb.values()) if cb else 0.0
    if mx > 0:
        for k in cb:
            cb[k] /= mx
    return cb


if __name__ == "__main__":
    import sys
    import osm, demand
    place = sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bengaluru"
    G, meta = osm.load_city(place=place, radius_m=900)
    bbox = osm.bbox_around(*meta["center"], 900)
    fac = fetch_facilities(bbox)
    care = map_to_junctions(G, fac)
    w = demand.junction_demand(G, points=demand.fetch_demand_points(bbox))
    dist = access_distances(G, care)
    served = served_fraction(G, care, w, dist=dist)
    crit = access_criticality(G, care, w)
    top = sorted(crit, key=crit.get, reverse=True)[:3]
    print(f"{place}: {len(fac)} facilities -> {len(care)} care junctions")
    print(f"demand within {DEFAULT_REACH_M} m of care: {served*100:.1f}%")
    print(f"top emergency-critical junctions: {[str(t) for t in top]}")
