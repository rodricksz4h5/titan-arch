"""
Titan Arch  --  synth.py

A reproducible synthetic road mask for demoing the gap-healing step without a
GPU or a downloaded satellite tile. Builds a road grid, then punches occlusion
gaps DIRECTLY on the roads -- the real-world pattern of a tree canopy or shadow
hiding a stretch of road.

Used by the dashboard's "reconstruction demo" and by the healing test.
"""

from __future__ import annotations

import numpy as np

# a default ~2 km tile over central Bengaluru (S, W, N, E)
DEFAULT_BBOX = (12.960, 77.580, 12.980, 77.600)


def bbox_around(lat, lon, half_m=1100):
    import math
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def grid_mask(size=256, spacing=40, seed=3, cuts=60, gap=3, bbox=None):
    """
    Returns (occluded_mask, clean_mask, bbox).
      occluded_mask : roads with occlusion gaps (the noisy mask a model produces)
      clean_mask    : the true road grid (ground truth, for comparison)

    Occlusion mimics tree canopy / shadow over road SEGMENTS (not intersections),
    which is the realistic case gap-healing is meant to recover.
    """
    H = W = size
    clean = np.zeros((H, W), bool)
    rows = list(range(spacing // 2, W, spacing))
    cols = list(range(spacing // 2, W, spacing))
    for r in rows:
        clean[max(0, r - 1):r + 2, :] = True
    for c in cols:
        clean[:, max(0, c - 1):c + 2] = True

    rng = np.random.default_rng(seed)
    occ = clean.copy()
    placed, tries = 0, 0
    while placed < cuts and tries < cuts * 25:
        tries += 1
        if rng.random() < 0.5:                          # cut a horizontal road
            r = rows[rng.integers(len(rows))]
            c = int(rng.integers(spacing, W - spacing))
            if min(abs(c - cc) for cc in cols) < 10:    # keep clear of junctions
                continue
        else:                                            # cut a vertical road
            c = cols[rng.integers(len(cols))]
            r = int(rng.integers(spacing, H - spacing))
            if min(abs(r - rr) for rr in rows) < 10:
                continue
        occ[max(0, r - gap):r + gap + 1, max(0, c - gap):c + gap + 1] = False
        placed += 1
    return occ, clean, (bbox if bbox else DEFAULT_BBOX)


if __name__ == "__main__":
    occ, clean, bbox = grid_mask()
    print(f"clean road px={int(clean.sum())}  occluded={int(occ.sum())}  bbox={bbox}")
    try:
        import imageio.v2 as imageio
        import os
        out = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(out, exist_ok=True)
        imageio.imwrite(os.path.join(out, "synth_occluded_mask.png"),
                        (occ * 255).astype("uint8"))
        imageio.imwrite(os.path.join(out, "synth_clean_mask.png"),
                        (clean * 255).astype("uint8"))
        print("wrote demo masks to data/")
    except Exception as e:  # noqa: BLE001
        print("(could not write png:", e, ")")
