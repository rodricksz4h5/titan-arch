"""
Titan Arch  --  elevation.py  (Layer 2 data: terrain height per junction)

Fetches ground elevation for every junction from a free, keyless DEM service
(OpenTopoData / SRTM 30 m, with Open-Elevation as a fallback). This is what lets
us simulate a flood: water finds the low ground, so junctions below a rising
water line fail TOGETHER -- a spatially-correlated, realistic disaster, not
random clicks.
"""

from __future__ import annotations

import time

import requests

UA = {"User-Agent": "TitanArch/0.1 (anuragp2003b@gmail.com)"}
OPENTOPO = "https://api.opentopodata.org/v1/srtm30m"
OPENELEV = "https://api.open-elevation.com/api/v1/lookup"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_elevations(G, batch=100, pause=1.05):
    """
    Returns {node_id: elevation_m} for every junction. Batched and rate-limited
    to respect the free tier. Falls back to Open-Elevation if OpenTopoData fails.
    """
    nodes = list(G)
    coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in nodes]
    elev = {}

    sess = requests.Session()
    ok = True
    for bi, chunk_idx in enumerate(_chunks(list(range(len(nodes))), batch)):
        locs = "|".join(f"{coords[i][0]},{coords[i][1]}" for i in chunk_idx)
        try:
            r = sess.get(OPENTOPO, params={"locations": locs}, headers=UA, timeout=40)
            r.raise_for_status()
            res = r.json()["results"]
            for i, rr in zip(chunk_idx, res):
                e = rr.get("elevation")
                if e is not None:
                    elev[nodes[i]] = float(e)
        except Exception:                       # noqa: BLE001
            ok = False
            break
        if bi:                                  # be polite to the free service
            time.sleep(pause)

    if not ok or len(elev) < 0.5 * len(nodes):  # fallback provider
        try:
            payload = {"locations": [{"latitude": la, "longitude": lo} for la, lo in coords]}
            r = sess.post(OPENELEV, json=payload, headers=UA, timeout=90)
            r.raise_for_status()
            for n, rr in zip(nodes, r.json()["results"]):
                if rr.get("elevation") is not None:
                    elev[n] = float(rr["elevation"])
        except Exception:                       # noqa: BLE001
            pass
    return elev


if __name__ == "__main__":
    import sys
    import osm
    place = sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bengaluru"
    G, meta = osm.load_city(place=place, radius_m=700)
    t = time.time()
    elev = fetch_elevations(G)
    vals = sorted(elev.values())
    print(f"{place}: {len(elev)}/{G.number_of_nodes()} junctions in {time.time()-t:.0f}s")
    if vals:
        print(f"elevation: min={vals[0]:.0f} m  median={vals[len(vals)//2]:.0f} m  max={vals[-1]:.0f} m"
              f"  (relief {vals[-1]-vals[0]:.0f} m)")
