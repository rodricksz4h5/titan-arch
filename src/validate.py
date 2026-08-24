"""
Titan Arch  --  validate.py  (Step 4: hard numbers)

Two evaluations:

  1. HEALING BENCHMARK (runs now, no GPU). Uses the synthetic generator, where
     we know the TRUE road grid. We occlude it, then check that our gap-healing
     reconstructs the real network -- and that the bridges it invents actually
     lie on roads the occlusion hid (healing precision against ground truth).

  2. MASK IoU (ready for the Colab output). Compares a predicted road mask
     against an OpenStreetMap-derived truth mask for the same area, with a
     normal IoU and a buffered "relaxed IoU" that forgives 1-2 px misalignment.

Run:  python validate.py
"""

from __future__ import annotations

import numpy as np

import mask_to_graph as m2g
import roadgraph as rg
import synth


# --------------------------------------------------------------------------- #
#  geo helpers
# --------------------------------------------------------------------------- #
def _ll_to_rc(lat, lon, shape, bbox):
    s, w, n, e = bbox
    h, wid = shape[:2]
    row = (n - lat) / (n - s) * (h - 1)
    col = (lon - w) / (e - w) * (wid - 1)
    return int(round(row)), int(round(col))


def _on_road(mask, r, c, tol=2):
    h, w = mask.shape
    r0, r1 = max(0, r - tol), min(h, r + tol + 1)
    c0, c1 = max(0, c - tol), min(w, c + tol + 1)
    return bool(mask[r0:r1, c0:c1].any())


# --------------------------------------------------------------------------- #
#  1. healing benchmark against synthetic ground truth
# --------------------------------------------------------------------------- #
def healing_benchmark(seed=3, cuts=60, max_gap_m=130, max_angle_deg=50, n_pairs=400):
    occ, clean, bbox = synth.grid_mask(seed=seed, cuts=cuts)

    G_truth, _ = m2g.mask_to_graph(clean, bbox, heal=False)
    G_raw, _ = m2g.mask_to_graph(occ, bbox, heal=False)
    G_heal, rep = m2g.mask_to_graph(occ, bbox, heal=True,
                                    max_gap_m=max_gap_m, max_angle_deg=max_angle_deg)

    def stats(G):
        c = rg.connectivity if False else None  # noqa
        comps = m2g.connectivity_stats(G)
        return comps

    s_truth = m2g.connectivity_stats(G_truth)
    s_raw = m2g.connectivity_stats(G_raw)
    s_heal = m2g.connectivity_stats(G_heal)

    # --- healing precision: do invented bridges fall on truly-occluded roads? ---
    correct, total = 0, 0
    for u, v, d in G_heal.edges(data=True):
        if not d.get("healed"):
            continue
        total += 1
        a = (G_heal.nodes[u]["lat"], G_heal.nodes[u]["lon"])
        b = (G_heal.nodes[v]["lat"], G_heal.nodes[v]["lon"])
        hits = 0
        N = 12
        for t in range(N + 1):
            lat = a[0] + (b[0] - a[0]) * t / N
            lon = a[1] + (b[1] - a[1]) * t / N
            r, c = _ll_to_rc(lat, lon, clean.shape, bbox)
            if _on_road(clean, r, c, tol=2):
                hits += 1
        if hits / (N + 1) >= 0.6:           # most of the bridge lies on a real road
            correct += 1
    precision = correct / total if total else 1.0

    # --- efficiency recovery toward the true network ---
    eff_truth = rg.global_efficiency(G_truth, sample=200)
    eff_raw = rg.global_efficiency(G_raw, sample=200)
    eff_heal = rg.global_efficiency(G_heal, sample=200)
    recov = (eff_heal - eff_raw) / (eff_truth - eff_raw) if eff_truth > eff_raw else 1.0

    # Connectivity Ratio (spec eval): % increase in the largest connected
    # component after the MST healing phase.
    lcc_before = s_raw["lcc_frac"] * s_raw["nodes"]
    lcc_after = s_heal["lcc_frac"] * s_heal["nodes"]
    conn_ratio = ((lcc_after - lcc_before) / lcc_before * 100) if lcc_before else 0.0

    return {
        "bbox": bbox,
        "truth": s_truth, "raw": s_raw, "healed": s_heal,
        "bridges_added": rep["bridges_added"],
        "healing_precision": precision,
        "connectivity_ratio": conn_ratio,
        "eff_truth": eff_truth, "eff_raw": eff_raw, "eff_heal": eff_heal,
        "efficiency_recovery": recov,
    }


# --------------------------------------------------------------------------- #
#  2. mask IoU (for the real Colab-produced mask vs OSM truth)
# --------------------------------------------------------------------------- #
def mask_iou(pred, truth):
    pred = np.asarray(pred) > 0
    truth = np.asarray(truth) > 0
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter / union) if union else 1.0


def occlusion_recall(pred, truth, visible):
    """Spec eval: recall of road pixels that were OCCLUDED (truth road where the
    input was hidden) -- i.e. roads recovered from under shadow/canopy.
    `visible` = the (possibly occluded) input road mask the model actually saw."""
    truth = np.asarray(truth) > 0
    pred = np.asarray(pred) > 0
    visible = np.asarray(visible) > 0
    occ = truth & ~visible                       # true road that was hidden
    denom = occ.sum()
    return float((pred & occ).sum() / denom) if denom else 1.0


def relaxed_iou(pred, truth, buffer_px=3):
    """IoU after dilating each mask by buffer_px (spec: 3-5 px tolerance), so a
    road predicted a few px off still counts as a hit (length-complete IoU)."""
    from scipy.ndimage import binary_dilation
    pred = np.asarray(pred) > 0
    truth = np.asarray(truth) > 0
    st = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), bool)
    pd, td = binary_dilation(pred, st), binary_dilation(truth, st)
    tp = np.logical_and(pred, td).sum() + np.logical_and(truth, pd).sum()
    fp = np.logical_and(pred, np.logical_not(td)).sum()
    fn = np.logical_and(truth, np.logical_not(pd)).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom else 1.0


def osm_truth_mask(bbox, shape):
    """Rasterise OSM roads for a bbox into a binary mask (the ground truth to
    score a predicted mask against). Needs network access (Overpass)."""
    import osm
    from skimage.draw import line
    s, w, n, e = bbox
    osm_json = osm.fetch_overpass((s, w, n, e))
    G = osm._simplify(osm._raw_graph(osm_json))
    h, wid = shape[:2]
    mask = np.zeros((h, wid), np.uint8)
    for u, v, d in G.edges(data=True):
        for (lon1, lat1), (lon2, lat2) in zip(d.get("geometry", []),
                                              d.get("geometry", [])[1:]):
            r0, c0 = _ll_to_rc(lat1, lon1, shape, bbox)
            r1, c1 = _ll_to_rc(lat2, lon2, shape, bbox)
            r0 = min(max(r0, 0), h - 1); r1 = min(max(r1, 0), h - 1)
            c0 = min(max(c0, 0), wid - 1); c1 = min(max(c1, 0), wid - 1)
            rr, cc = line(r0, c0, r1, c1)
            mask[rr, cc] = 255
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask > 0, np.ones((3, 3), bool)).astype(np.uint8) * 255
    return mask


# --------------------------------------------------------------------------- #
#  3. topological accuracy: avg-path-length error vs OSM ground truth
# --------------------------------------------------------------------------- #
def topological_accuracy(G_model, G_osm, n_pairs=200, seed=0):
    """
    Spec eval: run shortest-path between random point pairs on the ground-truth
    OSM graph vs the model's graph and measure the average-path-length error.
    Points are matched between graphs by nearest junction (so the two graphs
    need not share node ids). Returns (mean_relative_error, n_pairs_used).
    """
    import math
    import random

    import networkx as nx
    from scipy.spatial import cKDTree

    m_nodes = list(G_model)
    if len(m_nodes) < 2 or G_osm.number_of_nodes() < 2:
        return None, 0
    lat0 = sum(G_osm.nodes[v]["lat"] for v in G_osm) / G_osm.number_of_nodes()

    def xy(G, v):
        return (G.nodes[v]["lon"] * 111_320.0 * math.cos(math.radians(lat0)),
                G.nodes[v]["lat"] * 111_320.0)

    tree = cKDTree([xy(G_model, v) for v in m_nodes])

    def nearest_model(v_osm):
        _, idx = tree.query(xy(G_osm, v_osm), k=1)
        return m_nodes[int(idx)]

    rng = random.Random(seed)
    osm_nodes = list(G_osm)
    errs = []
    for _ in range(n_pairs * 3):
        if len(errs) >= n_pairs:
            break
        a, b = rng.sample(osm_nodes, 2)
        try:
            d_osm = nx.shortest_path_length(G_osm, a, b, weight="length")
            d_mod = nx.shortest_path_length(G_model, nearest_model(a),
                                            nearest_model(b), weight="length")
        except nx.NetworkXNoPath:
            continue
        if d_osm > 0:
            errs.append(abs(d_mod - d_osm) / d_osm)
    return (sum(errs) / len(errs) if errs else None), len(errs)


# --------------------------------------------------------------------------- #
#  report
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 64)
    print("TITAN ARCH — HEALING BENCHMARK (synthetic ground truth)")
    print("=" * 64)
    r = healing_benchmark()
    t, raw, heal = r["truth"], r["raw"], r["healed"]
    print(f"true road grid:     {t['nodes']} nodes, {t['components']} piece(s), "
          f"largest {t['lcc_frac']*100:.0f}%")
    print(f"after occlusion:    {raw['components']} pieces, "
          f"largest {raw['lcc_frac']*100:.0f}%   <- broken mask")
    print(f"after Titan heal:   {heal['components']} pieces, "
          f"largest {heal['lcc_frac']*100:.0f}%   <- reconstructed")
    print("-" * 64)
    print(f"healing method:          MST + Disjoint-Set (per spec)")
    print(f"bridges invented:        {r['bridges_added']}")
    print(f"healing PRECISION:       {r['healing_precision']*100:.1f}%  "
          f"(bridges that lie on real, occluded roads)")
    print(f"CONNECTIVITY RATIO:      +{r['connectivity_ratio']:.0f}%  "
          f"(growth of largest connected component after healing)")
    print(f"routing efficiency:      raw {r['eff_raw']:.2e} -> healed {r['eff_heal']:.2e}  "
          f"(truth {r['eff_truth']:.2e})")
    print(f"efficiency RECOVERY:     {r['efficiency_recovery']*100:.0f}%  "
          f"of the gap back to the true network")
    print("=" * 64)
    # eval-metric sanity checks
    occ, clean, bbox = synth.grid_mask()
    print(f"IoU sanity (clean vs clean):        {mask_iou(clean, clean):.3f}  "
          f"| relaxed(3px) = {relaxed_iou(clean, clean):.3f}")
    print(f"Occlusion-Recall sanity (clean recovers all): "
          f"{occlusion_recall(clean, clean, occ):.3f}")
    G_truth, _ = m2g.mask_to_graph(clean, bbox, heal=False)
    err, npairs = topological_accuracy(G_truth, G_truth, n_pairs=100)
    print(f"Topological-Accuracy sanity (graph vs itself): "
          f"APL error = {err*100:.2f}%  over {npairs} pairs")
