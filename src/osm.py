"""
Titan Arch  --  osm.py  (Step 1: real road networks)

Pulls a real city's drivable road network straight from OpenStreetMap via the
public Overpass API (no API key, no osmnx dependency) and turns it into the
same networkx graph the rest of Titan Arch analyses.

Key step: we SIMPLIFY the raw OSM data. Raw OSM roads are full of "shape
points" (degree-2 nodes that just trace a curve). For network analysis we only
care about intersections and dead-ends, so we contract every chain of shape
points into a single edge -- keeping the true road length and the polyline
geometry (so the map still draws the real curve).
"""

from __future__ import annotations

import time

import networkx as nx
import requests

from roadgraph import haversine_m

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "TitanArch/0.1 (anuragp2003b@gmail.com)"}

# road classes we treat as the drivable network
DRIVE = ("motorway|trunk|primary|secondary|tertiary|unclassified|"
         "residential|living_street|motorway_link|trunk_link|primary_link|"
         "secondary_link|tertiary_link|road")

# road hierarchy rank (bigger = higher capacity); used to pick a chain's class
_HW_RANK = {"motorway": 6, "trunk": 5, "primary": 4, "secondary": 3,
            "tertiary": 2, "unclassified": 1, "residential": 1,
            "living_street": 0, "road": 1}


# --------------------------------------------------------------------------- #
#  geocoding: place name -> bounding box
# --------------------------------------------------------------------------- #
def geocode_bbox(place):
    """Return (south, west, north, east) for a place name, via Nominatim."""
    r = requests.get(NOMINATIM, params={"q": place, "format": "json", "limit": 1},
                     headers=UA, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError(f"could not geocode '{place}'")
    s, n, w, e = (float(x) for x in js[0]["boundingbox"])  # nominatim: S,N,W,E
    return (s, w, n, e), (float(js[0]["lat"]), float(js[0]["lon"])), js[0].get("display_name", place)


def bbox_around(lat, lon, radius_m=1200):
    """A square bbox of +/- radius_m around a point."""
    dlat = radius_m / 111_320.0
    import math
    dlon = radius_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


# --------------------------------------------------------------------------- #
#  fetch raw roads from Overpass
# --------------------------------------------------------------------------- #
def fetch_overpass(bbox, retries=3):
    s, w, n, e = bbox
    q = f"""
    [out:json][timeout:90];
    (way["highway"~"{DRIVE}"]({s},{w},{n},{e}););
    (._;>;);
    out body;
    """
    last = None
    for i in range(retries):
        try:
            r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as ex:               # noqa: BLE001
            last = ex
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"Overpass failed after {retries} tries: {last}")


# --------------------------------------------------------------------------- #
#  raw OSM -> simplified networkx graph
# --------------------------------------------------------------------------- #
def _raw_graph(osm_json):
    """Build the full (un-simplified) graph: every OSM node + way segment."""
    nodes = {}
    for el in osm_json["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])

    Gf = nx.Graph()
    for nid, (la, lo) in nodes.items():
        Gf.add_node(nid, lat=la, lon=lo)

    for el in osm_json["elements"]:
        if el["type"] != "way":
            continue
        hw = el.get("tags", {}).get("highway", "residential")
        nd = [n for n in el.get("nodes", []) if n in nodes]
        for a, b in zip(nd, nd[1:]):
            if a == b:
                continue
            la1, lo1 = nodes[a]
            la2, lo2 = nodes[b]
            Gf.add_edge(a, b, length=haversine_m(la1, lo1, la2, lo2), highway=hw)
    # drop nodes that ended up with no edges
    Gf.remove_nodes_from([n for n in list(Gf) if Gf.degree(n) == 0])
    return Gf


def _simplify(Gf):
    """
    Contract chains of degree-2 nodes into single edges. Control nodes =
    intersections + dead-ends (degree != 2). Edge keeps summed length and the
    full polyline as 'geometry' = [[lon,lat], ...].
    """
    control = [n for n in Gf if Gf.degree(n) != 2]
    if not control:                       # pure loop fallback: keep everything
        control = list(Gf)

    Gs = nx.Graph()
    for c in control:
        Gs.add_node(c, lat=Gf.nodes[c]["lat"], lon=Gf.nodes[c]["lon"])

    visited_edges = set()
    for c in control:
        for nb in Gf.neighbors(c):
            if (c, nb) in visited_edges:
                continue
            # walk the chain c -> nb -> ... until next control node
            prev, cur = c, nb
            length = Gf[c][nb]["length"]
            hw_classes = [Gf[c][nb].get("highway", "residential")]
            geom = [[Gf.nodes[c]["lon"], Gf.nodes[c]["lat"]],
                    [Gf.nodes[nb]["lon"], Gf.nodes[nb]["lat"]]]
            visited_edges.add((c, nb))
            visited_edges.add((nb, c))
            while Gf.degree(cur) == 2 and cur not in control:
                nxts = [x for x in Gf.neighbors(cur) if x != prev]
                if not nxts:
                    break
                nn = nxts[0]
                length += Gf[cur][nn]["length"]
                hw_classes.append(Gf[cur][nn].get("highway", "residential"))
                geom.append([Gf.nodes[nn]["lon"], Gf.nodes[nn]["lat"]])
                visited_edges.add((cur, nn))
                visited_edges.add((nn, cur))
                prev, cur = cur, nn
            # the chain's class = its biggest road (highest capacity)
            hw = max(hw_classes, key=lambda h: _HW_RANK.get(h, 0))
            if cur != c and cur in Gs:      # close the simplified edge
                if Gs.has_edge(c, cur):
                    if length < Gs[c][cur]["length"]:
                        Gs[c][cur].update(length=length, geometry=geom, highway=hw)
                else:
                    Gs.add_edge(c, cur, length=length, geometry=geom, highway=hw)
    Gs.remove_nodes_from([n for n in list(Gs) if Gs.degree(n) == 0])
    return Gs


def load_city(place=None, lat=None, lon=None, radius_m=1200):
    """
    Main entry: returns (G, meta). Either pass a place name (geocoded) or an
    explicit lat/lon centre. radius_m controls the area pulled.
    """
    if place and lat is None:
        bbox, center, name = geocode_bbox(place)
        # clamp huge admin bboxes to a sane radius around the centre
        bbox = bbox_around(center[0], center[1], radius_m)
    else:
        center = (lat, lon)
        name = place or f"{lat:.4f},{lon:.4f}"
        bbox = bbox_around(lat, lon, radius_m)

    osm = fetch_overpass(bbox)
    Gf = _raw_graph(osm)
    Gs = _simplify(Gf)

    # keep only the largest connected piece for clean routing/analysis
    if Gs.number_of_nodes():
        biggest = max(nx.connected_components(Gs), key=len)
        Gs = Gs.subgraph(biggest).copy()

    meta = {"name": f"{name.split(',')[0]} (OSM)", "source": "OpenStreetMap",
            "center": [center[0], center[1]], "radius_m": radius_m,
            "raw_nodes": Gf.number_of_nodes()}
    return Gs, meta


if __name__ == "__main__":
    import sys
    place = sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bengaluru"
    rad = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print(f"fetching {place} (r={rad}m) ...")
    G, meta = load_city(place=place, radius_m=rad)
    print(f"raw nodes={meta['raw_nodes']}  ->  simplified "
          f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
