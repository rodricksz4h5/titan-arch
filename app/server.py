"""
Titan Arch  --  server.py

Flask backend for the dashboard. Holds one road network in memory and exposes:
  GET  /                  -> the dashboard
  GET  /api/graph         -> network as geojson + criticality + resilience summary
  POST /api/ablate        -> close junction(s); returns efficiency hit + reroutes
  POST /api/load_demo     -> (re)load the tiny hand-made map  (Step 0)

Step 1 will add /api/load_osm to pull a real city.
"""
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import roadgraph as rg  # noqa: E402

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")

app = Flask(__name__, static_folder=None)

# in-memory state ----------------------------------------------------------- #
STATE = {"G": None, "bc": None, "res": None, "sample": None, "meta": {},
         "recon": None, "weight": None, "elev": None, "flood_ctx": None,
         "care": None, "facilities": None, "served": None}


def _auto_sample(G):
    """Big networks: approximate betweenness/efficiency from a sample so the
    dashboard stays responsive. Small ones: exact."""
    n = G.number_of_nodes()
    return None if n <= 400 else max(200, n // 6)


def _analyse(G, sample=None, weight=None):
    if sample is None:
        sample = _auto_sample(G)
    res = rg.resilience_stress_test(G, top_k=10, sample=sample, weight=weight)
    STATE.update(G=G, bc=res["betweenness"], res=res, sample=sample, weight=weight,
                 elev=None, flood_ctx=None,            # invalidate flood cache
                 care=None, facilities=None, served=None, em_crit=None,  # + emergency
                 congestion=None)                      # + congestion cache
    return res


def _emergency_payload():
    if not STATE.get("care"):
        return None
    G = STATE["G"]
    crit = STATE.get("em_crit", {})
    top = sorted(crit, key=crit.get, reverse=True)[:6]
    return {
        "n_facilities": len(STATE["facilities"]),
        "served_fraction": STATE["served"],
        "facilities": [{"lat": f["lat"], "lon": f["lon"], "kind": f["kind"]}
                       for f in STATE["facilities"]],
        "critical": [{"id": str(n), "score": crit.get(n, 0.0),
                      "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]} for n in top],
    }


def _graph_payload():
    G, bc, res = STATE["G"], STATE["bc"], STATE["res"]
    gj = rg.to_geojson(G, bc)
    ranked = res["ranked_nodes"][:10]
    return {
        "meta": STATE["meta"],
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "geojson": gj,
        "gatekeepers": [{"id": str(n), "criticality": bc.get(n, 0.0),
                         "demand": round(STATE["weight"].get(n, 0.0), 1) if STATE["weight"] else None,
                         "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]}
                        for n in ranked],
        "demand_weighted": bool(STATE["weight"]),
        "emergency": _emergency_payload(),
        "congestion": ({"pct_overloaded": STATE["congestion"]["pct_overloaded"],
                        "vc": STATE["congestion"]["vc"],
                        "worst": STATE["congestion"]["worst"]}
                       if STATE.get("congestion") else None),
        "resilience_index": res["resilience_index"],
        "resilience_index_apl": res.get("resilience_index_apl"),
        "base_efficiency": res["base_efficiency"],
        "reconstruction": STATE.get("recon"),
        "stress_curve": [{"step": i + 1, "removed": str(s["removed"]),
                          "eff_ratio": s["eff_ratio"], "lcc_frac": s["lcc_frac"]}
                         for i, s in enumerate(res["steps"])],
        "stress_order": [str(n) for n in res["ranked_nodes"][:25]],
    }


# routes -------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC, fname)


def _init_demo():
    G = rg.demo_graph()
    STATE["meta"] = {"name": "Demo: two clusters + one bridge",
                     "source": "hand-made", "center": [12.9695, 77.5930]}
    STATE["recon"] = None
    _analyse(G)


@app.route("/api/load_demo", methods=["POST", "GET"])
def load_demo():
    _init_demo()
    return jsonify(_graph_payload())


@app.route("/api/load_osm", methods=["POST"])
def load_osm():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import osm  # imported lazily so the app still runs offline for the demo
    data = request.get_json(force=True) or {}
    place = (data.get("place") or "").strip()
    radius = int(data.get("radius_m", 1100))
    try:
        if place:
            G, meta = osm.load_city(place=place, radius_m=radius)
        else:
            G, meta = osm.load_city(lat=float(data["lat"]), lon=float(data["lon"]),
                                    radius_m=radius)
    except Exception as ex:  # noqa: BLE001
        return jsonify({"error": f"Could not load '{place or 'location'}': {ex}"}), 400
    if G.number_of_nodes() < 3:
        return jsonify({"error": "Too few roads found here; try a larger radius."}), 400

    # Layer 1: demand weighting (best-effort; falls back to uniform on failure)
    weight = None
    if data.get("demand", True):
        try:
            import demand as dmd
            s, w, n, e = (meta["center"][0] - 0.011, meta["center"][1] - 0.011,
                          meta["center"][0] + 0.011, meta["center"][1] + 0.011)
            pts = dmd.fetch_demand_points((s, w, n, e))
            weight = dmd.junction_demand(G, points=pts)
            meta["demand_points"] = len(pts)
        except Exception:
            weight = None
    meta["demand_weighted"] = weight is not None

    STATE["meta"] = meta
    STATE["recon"] = None
    _analyse(G, weight=weight)

    # Layer 3: emergency-care access (best-effort)
    STATE["care"] = STATE["facilities"] = STATE["served"] = None
    if data.get("emergency", True):
        try:
            import emergency as em
            s, w, n, e = (meta["center"][0] - 0.011, meta["center"][1] - 0.011,
                          meta["center"][0] + 0.011, meta["center"][1] + 0.011)
            fac = em.fetch_facilities((s, w, n, e))
            care = em.map_to_junctions(G, fac)
            if care:
                dist = em.access_distances(G, care)
                STATE["care"] = care
                STATE["facilities"] = fac
                STATE["served"] = em.served_fraction(G, care, weight, dist=dist)
                STATE["em_crit"] = em.access_criticality(G, care, weight)
        except Exception:
            import traceback
            traceback.print_exc()

    # Layer 4 congestion is computed lazily (on first heatmap toggle) to keep
    # the load fast -- see /api/congestion.
    return jsonify(_graph_payload())


@app.route("/api/congestion", methods=["POST"])
def congestion_route():
    G = STATE["G"]
    if G is None or STATE["meta"].get("source") != "OpenStreetMap":
        return jsonify({"error": "Load a real city first."}), 400
    if not STATE.get("congestion"):
        try:
            import capacity as capm
            STATE["congestion"] = capm.congestion(G, weight=STATE["weight"], sample=120)
        except Exception as ex:  # noqa: BLE001
            return jsonify({"error": f"Congestion model failed: {ex}"}), 400
    c = STATE["congestion"]
    return jsonify({"pct_overloaded": c["pct_overloaded"], "vc": c["vc"],
                    "worst": c["worst"]})


@app.route("/api/load_mask", methods=["POST"])
def load_mask():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import numpy as np
    import mask_to_graph as m2g
    data = request.get_json(force=True) or {}

    if data.get("demo") or not data.get("mask_b64"):
        import synth
        ctr = data.get("center")
        bb = synth.bbox_around(float(ctr[0]), float(ctr[1])) if ctr else None
        mask, _clean, bbox = synth.grid_mask(bbox=bb)
        where = f" over {data.get('place')}" if data.get("place") else ""
        name = f"Reconstruction demo (synthetic occluded mask{where})"
    else:
        import base64, io
        raw = base64.b64decode(data["mask_b64"].split(",")[-1])
        name = data.get("name", "Uploaded road mask")
        import geotiff
        if geotiff.is_geotiff(name, raw):
            # georeferenced product (LISS-IV / Cartosat / Sentinel-2): auto bbox
            mask, bbox = geotiff.read_geotiff_bytes(raw)
            name = data.get("name", "GeoTIFF mask") + " (auto-georeferenced)"
        else:
            import imageio.v2 as imageio
            if not data.get("bbox") or len(data["bbox"]) != 4:
                return jsonify({"error": "A plain image mask needs a bbox [S,W,N,E]; "
                                         "or upload a GeoTIFF and the bounds are read automatically."}), 400
            mask = np.asarray(imageio.imread(io.BytesIO(raw)))
            bbox = tuple(float(x) for x in data["bbox"])

    heal_kw = {"max_gap_m": int(data.get("max_gap_m", 130)),
               "max_angle_deg": int(data.get("max_angle_deg", 50))}
    G, rep = m2g.mask_to_graph(mask, bbox, heal=True, **heal_kw)
    if G.number_of_nodes() < 3:
        return jsonify({"error": "Mask produced too few roads."}), 400

    s, w, n, e = bbox
    STATE["meta"] = {"name": name, "source": "road mask",
                     "center": [(s + n) / 2, (w + e) / 2]}
    STATE["recon"] = rep
    _analyse(G)
    return jsonify(_graph_payload())


@app.route("/api/graph")
def graph():
    if STATE["G"] is None:
        _init_demo()
    return jsonify(_graph_payload())


@app.route("/api/recommend_repairs", methods=["POST", "GET"])
def recommend_repairs():
    if STATE["G"] is None:
        return jsonify({"error": "no graph loaded"}), 400
    ri_before = STATE["res"]["resilience_index"] if STATE["res"] else None
    out = rg.recommend_repairs(STATE["G"], top_n=3, sample=STATE["sample"],
                               ri_before=ri_before, weight=STATE["weight"])
    return jsonify(out)


@app.route("/api/flood_prepare", methods=["POST"])
def flood_prepare():
    G = STATE["G"]
    if G is None or STATE["meta"].get("source") != "OpenStreetMap":
        return jsonify({"error": "Load a real city first (flood needs terrain)."}), 400
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import elevation
    import flood
    if not STATE.get("elev"):
        STATE["elev"] = elevation.fetch_elevations(G)
    if len(STATE["elev"]) < 3:
        return jsonify({"error": "Could not read terrain elevation."}), 400
    sources = rg._pick_sources(G, min(200, G.number_of_nodes()))
    base_flow = rg._flow_sum(G, sources, STATE["weight"]) or 1e-12
    base_demand = sum(STATE["weight"].values()) if STATE["weight"] else G.number_of_nodes()
    STATE["flood_ctx"] = {"sources": sources, "base_flow": base_flow, "base_demand": base_demand}
    fc = flood.flood_curve(G, STATE["elev"], weight=STATE["weight"], steps=12)
    return jsonify(fc)


@app.route("/api/flood", methods=["POST"])
def flood_step():
    G = STATE["G"]
    if G is None or not STATE.get("elev"):
        return jsonify({"error": "Run flood_prepare first."}), 400
    import flood
    depth = float((request.get_json(force=True) or {}).get("depth_m", 0))
    ctx = STATE.get("flood_ctx") or {}
    sc = flood.flood_scenario(G, STATE["elev"], depth, weight=STATE["weight"],
                              sources=ctx.get("sources"), base_flow=ctx.get("base_flow"),
                              base_demand=ctx.get("base_demand"),
                              care_nodes=STATE.get("care"))
    return jsonify(sc)


@app.route("/api/ablate", methods=["POST"])
def ablate():
    if STATE["G"] is None:
        return jsonify({"error": "no graph loaded"}), 400
    data = request.get_json(force=True) or {}
    dead = [_coerce(STATE["G"], x) for x in data.get("dead", [])]
    pairs = [(_coerce(STATE["G"], s), _coerce(STATE["G"], t))
             for s, t in data.get("pairs", [])]
    out = rg.ablate(STATE["G"], dead, probe_pairs=pairs, sample=STATE["sample"],
                    weight=STATE["weight"])
    if data.get("congestion") and dead and STATE.get("congestion"):
        try:
            import capacity as capm
            out["congestion"] = capm.congestion_under_failure(
                STATE["G"], dead, weight=STATE["weight"], sample=120,
                scale=STATE["congestion"]["scale"])
        except Exception:
            out["congestion"] = None
    # attach path coords for drawing the reroute
    for trip in out["trips"]:
        p = trip.get("new_path")
        if p:
            trip["new_path_coords"] = [[STATE["G"].nodes[n]["lon"],
                                        STATE["G"].nodes[n]["lat"]] for n in p]
        trip["src"] = str(trip["src"])
        trip["dst"] = str(trip["dst"])
    out["closed"] = [str(x) for x in out["closed"]]
    return jsonify(out)


def _coerce(G, x):
    """Node ids may be ints (demo) or strings (osm). Match whatever G uses."""
    if x in G:
        return x
    for cand in (int(x) if str(x).lstrip("-").isdigit() else None, str(x)):
        if cand is not None and cand in G:
            return cand
    return x


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    _init_demo()
    print(f"Titan Arch running on http://127.0.0.1:{a.port}")
    app.run(host="127.0.0.1", port=a.port, debug=False)
