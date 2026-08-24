"""
Titan Arch  --  demand.py  (Layer 1: real travel demand)

Turns free OpenStreetMap data into a per-junction "demand" weight, so the
analysis reflects where people actually are -- not an abstract assumption that
every junction matters equally.

Demand proxy = local built-up activity: we pull building footprints (and key
trip-generators like schools, hospitals, markets, offices) around the area and
assign each one to its nearest junction. A junction surrounded by dense
buildings carries high demand; one in an empty patch carries little.

This single weight upgrades every downstream number: criticality becomes
"how many real trips depend on this junction", not "how many abstract paths".
"""

from __future__ import annotations

import math

import numpy as np
import requests
from scipy.spatial import cKDTree

from osm import OVERPASS, UA   # reuse the same endpoint + user-agent

# trip-generators worth extra weight (people travel TO these)
POI_WEIGHT = 6.0
POI_FILTER = ('["amenity"~"hospital|clinic|school|college|university|marketplace|'
              'bus_station|community_centre"]')


def _to_xy(lat, lon, lat0):
    return (lon * 111_320.0 * math.cos(math.radians(lat0)), lat * 111_320.0)


def fetch_demand_points(bbox, retries=3):
    """Return (lat, lon, weight) for buildings (+ weighted POIs) in the bbox."""
    s, w, n, e = bbox
    q = f"""
    [out:json][timeout:90];
    (
      way["building"]({s},{w},{n},{e});
      relation["building"]({s},{w},{n},{e});
      node{POI_FILTER}({s},{w},{n},{e});
      way{POI_FILTER}({s},{w},{n},{e});
    );
    out center;
    """
    import time
    last = None
    for i in range(retries):
        try:
            r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=120)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 + 3 * i)
    else:
        raise RuntimeError(f"Overpass demand fetch failed: {last}")

    pts = []
    for el in js["elements"]:
        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif "lat" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            continue
        is_poi = "amenity" in el.get("tags", {})
        pts.append((lat, lon, POI_WEIGHT if is_poi else 1.0))
    return pts


def junction_demand(G, bbox=None, points=None):
    """
    Assign each junction a demand weight = total weight of the buildings/POIs
    that are closest to it. Returns {node_id: weight} (>=1 everywhere via a
    small floor so isolated junctions still participate).
    """
    nodes = list(G)
    if not nodes:
        return {}
    if points is None:
        points = fetch_demand_points(bbox)

    lat0 = sum(G.nodes[v]["lat"] for v in nodes) / len(nodes)
    node_xy = np.array([_to_xy(G.nodes[v]["lat"], G.nodes[v]["lon"], lat0) for v in nodes])
    tree = cKDTree(node_xy)

    weight = {v: 0.0 for v in nodes}
    if points:
        pxy = np.array([_to_xy(p[0], p[1], lat0) for p in points])
        pw = np.array([p[2] for p in points])
        _, idx = tree.query(pxy, k=1)
        for j, w in zip(idx, pw):
            weight[nodes[j]] += float(w)

    # floor so every junction has a little baseline demand
    floor = max(1.0, 0.02 * (sum(weight.values()) / max(1, len(nodes))))
    for v in nodes:
        weight[v] += floor
    return weight


if __name__ == "__main__":
    import sys
    import osm
    place = sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bengaluru"
    G, meta = osm.load_city(place=place, radius_m=900)
    pts = fetch_demand_points(osm.bbox_around(*meta["center"], 900))
    w = junction_demand(G, points=pts)
    vals = sorted(w.values(), reverse=True)
    print(f"{place}: {G.number_of_nodes()} junctions, {len(pts)} demand points")
    print(f"demand weight: max={vals[0]:.1f}  median={vals[len(vals)//2]:.1f}  min={vals[-1]:.2f}")
    top = sorted(w, key=w.get, reverse=True)[:3]
    print("highest-demand junctions:", [(str(t), round(w[t], 1)) for t in top])
