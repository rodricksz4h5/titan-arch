"""
Titan Arch  --  validate_real.py

The real-data validation harness: it takes a road mask produced from ACTUAL
satellite imagery (the Colab model's output) for a real place, fetches the
OpenStreetMap road network as ground truth for the same footprint, and scores
the full evaluation suite the brief asks for:

  * pixel:   IoU, Dice, Precision/Recall, Relaxed-IoU (3 px tolerance)
  * graph:   Connectivity Ratio (LCC growth after MST/DSU healing)
  * routing: Topological Accuracy (average-path-length error vs OSM)

It prints a report card and writes report_card.json / report_card.md.

Usage:
  # real model mask (GeoTIFF auto-reads bounds, or pass --bbox S W N E):
  python validate_real.py path/to/road_mask.tif --place "Indiranagar, Bengaluru"

  # dry-run with no model yet (uses OSM, degrades it to stand in for a prediction
  # so you can see the harness produce real numbers today):
  python validate_real.py --demo --place "Indiranagar, Bengaluru"
"""

from __future__ import annotations

import json
import sys

import numpy as np
from skimage.draw import line
from scipy.ndimage import binary_dilation

import osm
import mask_to_graph as m2g
import validate as V


# --------------------------------------------------------------------------- #
#  OSM ground truth (fetched once; used for both the truth mask and graph)
# --------------------------------------------------------------------------- #
def osm_ground_truth(bbox, shape):
    osm_json = osm.fetch_overpass(bbox)
    G = osm._simplify(osm._raw_graph(osm_json))
    if G.number_of_nodes():
        import networkx as nx
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mask = _rasterise(G, shape, bbox)
    return mask, G


def _rasterise(G, shape, bbox, width_px=3):
    h, wid = shape[:2]
    mask = np.zeros((h, wid), np.uint8)
    for u, v, d in G.edges(data=True):
        pts = d.get("geometry") or [
            [G.nodes[u]["lon"], G.nodes[u]["lat"]], [G.nodes[v]["lon"], G.nodes[v]["lat"]]]
        for (lon1, lat1), (lon2, lat2) in zip(pts, pts[1:]):
            r0, c0 = V._ll_to_rc(lat1, lon1, shape, bbox)
            r1, c1 = V._ll_to_rc(lat2, lon2, shape, bbox)
            r0 = min(max(r0, 0), h - 1); r1 = min(max(r1, 0), h - 1)
            c0 = min(max(c0, 0), wid - 1); c1 = min(max(c1, 0), wid - 1)
            rr, cc = line(r0, c0, r1, c1)
            mask[rr, cc] = 255
    return binary_dilation(mask > 0, np.ones((width_px, width_px), bool)).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
#  the evaluation
# --------------------------------------------------------------------------- #
def evaluate(pred_mask, bbox):
    pred = np.asarray(pred_mask)
    if pred.ndim == 3:
        pred = pred[..., 0]
    truth_mask, G_osm = osm_ground_truth(bbox, pred.shape)

    pb, tb = pred > 0, truth_mask > 0
    tp = int(np.logical_and(pb, tb).sum())
    fp = int(np.logical_and(pb, ~tb).sum())
    fn = int(np.logical_and(~pb, tb).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    pix = {
        "IoU": V.mask_iou(pred, truth_mask),
        "Dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "Precision": prec, "Recall": rec,
        "F1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "Relaxed_IoU_3px": V.relaxed_iou(pred, truth_mask, buffer_px=3),
    }
    # (Occlusion-Recall needs known occlusion -> measured in the synthetic
    #  benchmark, validate.py; it isn't recoverable from a real image.)

    # graph: build from prediction, heal, measure connectivity gain + topo accuracy
    G_raw, _ = m2g.mask_to_graph(pred, bbox, heal=False)
    G_heal, rep = m2g.mask_to_graph(pred, bbox, heal=True,
                                    max_gap_m=130, max_angle_deg=50)
    sb, sa = rep["before"], rep["after"]
    lcc_b = sb["lcc_frac"] * sb["nodes"]
    lcc_a = sa["lcc_frac"] * sa["nodes"]
    apl_err, npairs = V.topological_accuracy(G_heal, G_osm, n_pairs=150)

    graph = {
        "pred_nodes": sa["nodes"], "osm_nodes": G_osm.number_of_nodes(),
        "components_before": sb["components"], "components_after": sa["components"],
        "bridges_added": rep["bridges_added"],
        "Connectivity_Ratio_pct": ((lcc_a - lcc_b) / lcc_b * 100) if lcc_b else 0.0,
        "Topological_APL_error_pct": (apl_err * 100) if apl_err is not None else None,
        "topo_pairs": npairs,
    }
    return {"bbox": bbox, "pixel": pix, "graph": graph}


# --------------------------------------------------------------------------- #
#  dry-run: stand in for a model by degrading OSM (so the harness runs today)
# --------------------------------------------------------------------------- #
def pseudo_prediction(bbox, size=1024, seed=5):
    """Rasterise OSM, then occlude + roughen it to imitate a real model output."""
    rng = np.random.default_rng(seed)
    _, G = osm_ground_truth(bbox, (size, size))
    clean = _rasterise(G, (size, size), bbox)
    pred = clean.copy()
    # occlude road stretches (tree/shadow) -> tests healing + recall
    on = np.argwhere(clean > 0)
    for _ in range(220):
        r, c = on[rng.integers(len(on))]
        s = int(rng.integers(2, 5))
        pred[max(0, r - s):r + s, max(0, c - s):c + s] = 0
    # a little false positive / negative speckle, like a real segmenter
    sp = rng.random(pred.shape)
    pred[sp > 0.9995] = 255
    pred[(clean > 0) & (sp > 0.97)] = 0
    return pred, clean


# --------------------------------------------------------------------------- #
def _report(res):
    p, g = res["pixel"], res["graph"]
    L = []
    L.append("=" * 60)
    L.append("TITAN ARCH — REAL-DATA VALIDATION REPORT CARD")
    L.append(f"footprint (S,W,N,E): {tuple(round(x,5) for x in res['bbox'])}")
    L.append("=" * 60)
    L.append("PIXEL (extracted roads vs OpenStreetMap truth)")
    for k in ("IoU", "Dice", "Precision", "Recall", "F1", "Relaxed_IoU_3px"):
        if k in p:
            L.append(f"  {k:<18} {p[k]*100:6.1f}%")
    L.append("-" * 60)
    L.append("GRAPH / ROUTING")
    L.append(f"  nodes (pred / OSM)   {g['pred_nodes']} / {g['osm_nodes']}")
    L.append(f"  components  {g['components_before']} -> {g['components_after']}  "
             f"(MST/DSU bridges: {g['bridges_added']})")
    L.append(f"  Connectivity Ratio   +{g['Connectivity_Ratio_pct']:.0f}%  (LCC growth after healing)")
    te = g["Topological_APL_error_pct"]
    L.append(f"  Topological Accuracy {('%.1f%% APL error' % te) if te is not None else 'n/a'}"
             f"  over {g['topo_pairs']} routes")
    L.append("=" * 60)
    return "\n".join(L)


if __name__ == "__main__":
    args = sys.argv[1:]
    demo = "--demo" in args
    place = None
    if "--place" in args:
        place = args[args.index("--place") + 1]
    bbox = None
    if "--bbox" in args:
        i = args.index("--bbox")
        bbox = tuple(float(x) for x in args[i + 1:i + 5])

    if demo:
        if not bbox:
            _bb, center, _name = osm.geocode_bbox(place or "Indiranagar, Bengaluru")
            bbox = osm.bbox_around(center[0], center[1], 900)
        print(f"[dry-run] standing in for the model by degrading OSM over {place}\n")
        pred, clean = pseudo_prediction(bbox)
        res = evaluate(pred, bbox)
    else:
        mask_path = args[0]
        import geotiff
        if geotiff.is_geotiff(mask_path):
            pred, bbox = geotiff.read_geotiff_path(mask_path)
        else:
            import imageio.v2 as imageio
            pred = np.asarray(imageio.imread(mask_path))
            if bbox is None:
                raise SystemExit("plain image needs --bbox S W N E (or pass a GeoTIFF)")
        res = evaluate(pred, bbox)

    report = _report(res)
    print(report)
    with open("report_card.json", "w") as f:
        json.dump(res, f, indent=2)
    with open("report_card.md", "w", encoding="utf-8") as f:
        f.write("```\n" + report + "\n```\n")
    print("\nwrote report_card.json / report_card.md")
