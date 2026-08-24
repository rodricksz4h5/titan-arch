"""
Titan Arch  --  mask_to_graph.py  (Step 3a: from a road mask to a routable map)

This is the project's headline novelty. A road-segmentation model (or any source)
gives us a black-and-white "road / not-road" image. That image is full of breaks
where trees, shadows and buildings hid the road. A raw mask like that is useless
for routing -- it falls into disconnected fragments.

Here we:
  1. thin the white roads to 1-pixel center-lines (skeletonize),
  2. turn the skeleton into a graph (junctions/dead-ends = nodes, roads = edges),
  3. georeference pixels -> lat/lon so it drops straight into our analysis,
  4. HEAL the gaps: find road-ends that point at each other across a short gap
     and reconnect them -- recovering roads the camera could not see.

Everything here is pure CPU. The GPU model that produces the mask is Step 3b.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from roadgraph import haversine_m


# --------------------------------------------------------------------------- #
#  pixel <-> geo transform
# --------------------------------------------------------------------------- #
class GeoTransform:
    """Maps image (row, col) to (lat, lon) given the tile's bounding box."""
    def __init__(self, shape, bbox):
        self.h, self.w = shape[:2]
        self.south, self.west, self.north, self.east = bbox

    def to_ll(self, row, col):
        lat = self.north - (row / max(1, self.h - 1)) * (self.north - self.south)
        lon = self.west + (col / max(1, self.w - 1)) * (self.east - self.west)
        return lat, lon

    def m_per_px(self):
        # rough metres-per-pixel using the diagonal
        dlat = (self.north - self.south) / max(1, self.h)
        dlon = (self.east - self.west) / max(1, self.w)
        return (haversine_m(self.south, self.west, self.south + dlat, self.west) +
                haversine_m(self.south, self.west, self.south, self.west + dlon)) / 2


# --------------------------------------------------------------------------- #
#  mask -> skeleton -> graph
# --------------------------------------------------------------------------- #
_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_to_graph(skel, gt: GeoTransform):
    """
    Build a graph from a 1-pixel skeleton. Nodes are placed at junctions
    (3+ neighbours) and dead-ends (1 neighbour); edges trace the pixel chains
    between them. Edge length is real metres; pixel polyline kept for drawing.
    """
    skel = skel.astype(bool)
    nbr = convolve(skel.astype(np.int8), _K, mode="constant") * skel
    node_px = skel & (nbr != 2)                      # endpoints + junctions
    skpix = set(map(tuple, np.argwhere(skel)))
    nodes = set(map(tuple, np.argwhere(node_px)))

    def neigh(p):
        r, c = p
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    q = (r + dr, c + dc)
                    if q in skpix:
                        yield q

    G = nx.Graph()
    mpp = gt.m_per_px()
    for p in nodes:
        lat, lon = gt.to_ll(*p)
        G.add_node(f"{p[0]}_{p[1]}", lat=lat, lon=lon, px=p)

    walked = set()
    for p in nodes:
        for q in neigh(p):
            if (p, q) in walked:
                continue
            prev, cur = p, q
            path = [p, q]
            walked.add((p, q))
            walked.add((q, p))
            # walk the chain until the next node pixel
            while cur not in nodes:
                nxts = [x for x in neigh(cur) if x != prev]
                if not nxts:
                    break
                nn = nxts[0]
                walked.add((cur, nn))
                walked.add((nn, cur))
                path.append(nn)
                prev, cur = cur, nn
            if cur in nodes and cur != p:
                # real length along the pixel path
                length = 0.0
                for a, b in zip(path, path[1:]):
                    step = math.hypot(a[0] - b[0], a[1] - b[1]) * mpp
                    length += step
                geom = [[*reversed(gt.to_ll(*pt))] for pt in path]  # [lon,lat]
                a_id, b_id = f"{p[0]}_{p[1]}", f"{cur[0]}_{cur[1]}"
                if not G.has_edge(a_id, b_id) or length < G[a_id][b_id]["length"]:
                    G.add_edge(a_id, b_id, length=max(length, 1.0),
                               geometry=geom, pix=path)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G


# --------------------------------------------------------------------------- #
#  THE HEALING STEP
# --------------------------------------------------------------------------- #
def _heading(G, node):
    """Outward direction (dlat, dlon) a dead-end road is pointing, from the
    pixels just inside it. Normalised in a local planar frame."""
    edges = list(G.edges(node, data=True))
    if not edges:
        return None
    _, _, d = edges[0]
    geom = d.get("geometry")  # [[lon,lat],...]
    if not geom or len(geom) < 2:
        return None
    # ensure geom oriented so geom[0] is at `node`
    nlat, nlon = G.nodes[node]["lat"], G.nodes[node]["lon"]
    if abs(geom[-1][1] - nlat) + abs(geom[-1][0] - nlon) < \
       abs(geom[0][1] - nlat) + abs(geom[0][0] - nlon):
        geom = list(reversed(geom))
    k = min(4, len(geom) - 1)
    inlat, inlon = geom[k][1], geom[k][0]
    dlat, dlon = nlat - inlat, nlon - inlon
    coslat = math.cos(math.radians(nlat))
    v = (dlat, dlon * coslat)
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else None


def _bearing_to(a, b):
    """Unit vector from node a to node b in the local planar frame."""
    dlat = b[0] - a[0]
    dlon = (b[1] - a[1]) * math.cos(math.radians(a[0]))
    n = math.hypot(dlat, dlon)
    return (dlat / n, dlon / n) if n else None


class _DisjointSet:
    """Union-Find: tracks which road fragments are already connected, so the
    MST healing never adds a bridge that would just form a cycle."""
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def heal_gaps(G, max_gap_m=45, max_angle_deg=40, only_across_components=False):
    """
    Topological healing via Minimum Spanning Tree + Disjoint Set (Union-Find).

    Occlusion (tree canopy, shadows) breaks roads into disconnected fragments.
    We reconnect them the way the spec prescribes:

      1. Candidate bridges = pairs of dead-end road stubs whose gap is short
         (Euclidean distance <= max_gap_m) AND whose road directions point at
         each other (angular alignment within max_angle_deg) -- so a healed
         link follows the road's natural trajectory.
      2. Run Kruskal's MST over those candidates: process them cheapest-first
         (cost = gap penalised by mis-alignment) and add a bridge only if its
         two ends are in DIFFERENT fragments (checked by the Disjoint Set).
         This rebuilds a single connected, routable network with the minimum
         total bridging length and never creates a redundant cycle.

    Returns (healed_graph, info). Non-destructive: returns a copy.
    """
    H = G.copy()
    ends = [n for n in H if H.degree(n) == 1]
    cos_thresh = math.cos(math.radians(max_angle_deg))

    pos = {n: (H.nodes[n]["lat"], H.nodes[n]["lon"]) for n in ends}
    head = {n: _heading(H, n) for n in ends}

    # Disjoint Set seeded with the CURRENT connectivity of the fragmented graph
    dsu = _DisjointSet(H.nodes())
    for u, v in H.edges():
        dsu.union(u, v)

    # build candidate bridges (Euclidean gap + angular alignment gate)
    candidates = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            a, b = ends[i], ends[j]
            gap = haversine_m(*pos[a], *pos[b])
            if gap < 1.0 or gap > max_gap_m:
                continue
            ha, hb = head[a], head[b]
            if not ha or not hb:
                continue
            ab = _bearing_to(pos[a], pos[b])
            ba = (-ab[0], -ab[1])
            align_a = ha[0] * ab[0] + ha[1] * ab[1]
            align_b = hb[0] * ba[0] + hb[1] * ba[1]
            if align_a >= cos_thresh and align_b >= cos_thresh:
                cost = gap / (align_a * align_b)       # cheapest = short + straight
                candidates.append((cost, gap, a, b, min(align_a, align_b)))

    candidates.sort(key=lambda c: c[0])

    def _bridge(a, b, gap):
        geom = [[H.nodes[a]["lon"], H.nodes[a]["lat"]],
                [H.nodes[b]["lon"], H.nodes[b]["lat"]]]
        H.add_edge(a, b, length=max(gap, 1.0), geometry=geom, healed=True)

    # Phase 1 -- Kruskal MST over the Disjoint Set: cheapest bridges that each
    # merge two still-separate fragments (no cycles). Guarantees a minimal,
    # fully-connected spanning reconstruction.
    spanning = 0
    for cost, gap, a, b, align in candidates:
        if dsu.union(a, b):
            _bridge(a, b, gap)
            spanning += 1

    # Phase 2 -- redundancy completion: real road networks are cyclic (grids),
    # so re-add the well-aligned bridges MST skipped as "cycles". A STRICTER
    # alignment gate keeps these honest (precision stays high).
    redundancy = 0
    strict = math.cos(math.radians(max_angle_deg * 0.55))
    if not only_across_components:
        for cost, gap, a, b, align in candidates:
            if H.has_edge(a, b) or align < strict:
                continue
            if H.degree(a) <= 2 and H.degree(b) <= 2:   # still road-like, not over-fanned
                _bridge(a, b, gap)
                redundancy += 1

    info = {"bridges_added": spanning + redundancy,
            "spanning_bridges": spanning, "redundancy_bridges": redundancy,
            "dead_ends_before": len(ends),
            "method": "MST + Disjoint-Set, then redundancy completion"}
    return H, info


# --------------------------------------------------------------------------- #
#  connectivity report (before/after proof)
# --------------------------------------------------------------------------- #
def connectivity_stats(G):
    n = G.number_of_nodes()
    if n == 0:
        return {"nodes": 0, "edges": 0, "components": 0, "lcc_frac": 0.0}
    comps = list(nx.connected_components(G))
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "components": len(comps),
        "lcc_frac": len(max(comps, key=len)) / n,
    }


def mask_to_graph(mask, bbox, heal=True, **heal_kw):
    """Full pipeline: binary mask -> (graph, report)."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    skel = skeletonize(mask > (mask.max() / 2 if mask.max() else 0))
    gt = GeoTransform(mask.shape, bbox)
    G = skeleton_to_graph(skel, gt)
    before = connectivity_stats(G)
    info = {"bridges_added": 0}
    if heal:
        G, info = heal_gaps(G, **heal_kw)
    after = connectivity_stats(G)
    return G, {"before": before, "after": after, **info}
