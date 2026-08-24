# Titan Arch

**Find the junctions a city can't live without — then simulate what happens when they fail.**

Titan Arch turns a city's road map into a *routable network*, finds the
intersections the whole city depends on ("gatekeepers"), scores how fragile the
city is (the **Resilience Index**), and lets a planner click any junction to
close it and instantly watch traffic re-route — or get cut off.

It's built for the *Route Resilience* problem: occlusion-robust road extraction
plus graph-theoretic criticality analysis for urban mobility.

---

## What it does today

1. **Load a real city** — type a place name; Titan Arch pulls the drivable road
   network live from OpenStreetMap (Overpass API, no key needed) and simplifies
   it into a clean graph of intersections and roads (keeping true road lengths
   and curves).
2. **Find the gatekeepers** — betweenness centrality ranks the junctions that
   the most cross-city trips are forced through.
3. **Score resilience** — a stress test removes the top gatekeepers one by one
   and measures how the city's overall efficiency collapses. The leftover
   efficiency is the **Resilience Index** (1.0 = robust, ~0 = fragile).
4. **Simulate failure interactively** — click any junction on the map to close
   it (flood, accident, construction). The dashboard shows the city-wide
   efficiency hit, how much the network fragments, and re-routes a sample
   cross-city trip (or flags it as cut off).

5. **Reconstruct roads from a satellite mask** — a road-segmentation model paints
   roads from a satellite photo, but trees/shadows leave breaks. Titan Arch thins
   the mask to centre-lines, builds a graph, and **heals the gaps** by reconnecting
   road-ends that point at each other across a short break — recovering roads the
   camera could not see. A before/after panel proves the connectivity gain
   (on a synthetic occluded grid: disconnected pieces 45→21, largest connected
   area 15%→64%, routing efficiency +37%). Try it with the **Reconstruction
   demo** button, or upload your own mask.

The segmentation model itself (the only GPU part) lives in
`colab/road_segmentation.ipynb` — fine-tune it on free Colab, export a road mask,
and upload it to the app.

Also: a **repair recommender** that proposes the single new road link which most
improves the Resilience Index (e.g. "add this 32 m link → 0.86 → 0.93").

**Layer 4 — capacity & congestion.** Every road gets a capacity from its OSM
class; travel demand is routed onto shortest paths (a demand-weighted traffic
assignment) to give each road a volume/capacity ratio, turned into a travel-time
penalty by the BPR function. Toggle the congestion heatmap (green = free → red =
jammed), then fail a junction and watch traffic **redistribute and pile up** —
roads jump e.g. V/C 1.4 → 3.0, overloaded road-km doubles, with a real
reroute-delay number. Computed lazily to keep load fast.

**Layer 2 — flood / hazard scenarios.** Each junction's ground elevation is
fetched from a free DEM (SRTM 30 m). A slider raises the water line; junctions
below it fail *together* (a realistic, spatially-correlated disaster, not random
clicks). The dashboard shows what the city has left at each depth — % junctions
flooded, **% of travel demand still connected**, efficiency — plus a flood-
tolerance curve. (e.g. for one area: +12 m floods 40% of junctions but cuts
connected demand to ~39%.)

**Layer 3 — emergency-access criticality.** Hospitals, clinics and fire stations
are pulled from OpenStreetMap and mapped onto the network. Titan Arch reports
what share of the population can reach care, and which junctions that access most
depends on (the "emergency gatekeepers" — usually *different* from the traffic
gatekeepers). Folded into the flood model: as water rises you see **% of people
who can still reach a hospital** collapse (e.g. +16 m → only 28% retain access).

**Layer 1 — demand weighting.** Criticality, resilience and repairs can be
weighted by *real travel demand* (building density + trip-generators like
hospitals/schools/markets, pulled from OpenStreetMap), so a junction's score
means "how many real trips depend on it", not "how many abstract paths". Toggle
"weight by population" when loading a city. We implement demand-weighted
betweenness (a weighted Brandes) ourselves, since networkx has no
endpoint-weighted version.

---

## Spec compliance (Route Resilience)

| Spec requirement | How Titan Arch meets it |
|---|---|
| Occlusion-robust **Transformer** segmentation | `colab/road_segmentation.ipynb`: U-Net baseline **and** SegFormer/MixVision transformer (`BACKBONE='mit_b0'`), with occlusion-simulation augmentation + combined Dice/IoU/boundary loss |
| Topological healing via **MST + Disjoint Set** (Euclidean + angular) | `mask_to_graph.heal_gaps`: Kruskal MST over a Union-Find, candidate bridges gated by gap distance + road-direction alignment |
| Gatekeepers via betweenness centrality | `roadgraph.criticality` (+ demand-weighted variant) |
| Node-ablation stress test + **Resilience Index = baseline APL / perturbed APL** | `roadgraph.resilience_stress_test` reports the spec's path-length ratio (`resilience_index_apl`) and an efficiency variant |
| Dashboard: heatmap, click-to-disable, **travel-time increase** | Leaflet UI: criticality-coloured roads, click-to-fail with reroute + `+min` |
| Eval: IoU, **Relaxed IoU (3 px)**, **Occlusion-Recall**, **Connectivity Ratio**, **Topological Accuracy (APL error)** | all in `validate.py` |
| Datasets: **LISS-IV / Cartosat-3 / Sentinel-2** (GeoTIFF), **rasterio/GDAL** stack | `geotiff.py`: drop a GeoTIFF, bounds auto-read (no manual coords); notebook reads/writes GeoTIFF too |

## Validation (the numbers)

**Gap-healing benchmark** (`python src/validate.py`) — run against synthetic
ground truth, where the true road grid is known, then occluded like a tree
canopy and reconstructed:

| | true grid | broken mask | after healing |
|---|---|---|---|
| connected pieces | 1 | 19 | **2** |
| largest connected area | 100% | 17% | **99%** |

- **Healing precision: 100%** — every bridge it invents lies on a real road the
  occlusion hid (it never fabricates a road that wasn't there).
- **Connectivity Ratio: +480%** — growth of the largest connected component
  after the MST/Disjoint-Set healing phase.
- **Routing efficiency recovery: ~49%** of the gap back to the true network.
- Healing method: **MST + Disjoint Set** (minimal spanning reconnection) **then a
  redundancy-completion pass** that restores the grid loops real road networks
  have (stricter angular gate to keep precision). Gap cost = Euclidean distance
  penalised by angular mis-alignment.

**Road-mask accuracy** (`validate.mask_iou`, `relaxed_iou`) — standard IoU plus a
buffered "relaxed IoU" (forgives 1–2 px misalignment) for scoring a predicted
mask against an OpenStreetMap-rasterised truth mask. Wired and unit-checked;
the live number comes from the mask the Colab notebook produces.

**Real-data validation harness** (`validate_real.py`) — the scientific backbone.
Feed it the road mask the Colab model produced from *actual* satellite imagery
for a real place; it fetches the OSM road network as ground truth for the same
footprint and prints a **report card** with the spec's full evaluation suite —
IoU, Dice, Precision/Recall, Relaxed-IoU (3 px), **Connectivity Ratio** (LCC
growth after MST/DSU healing) and **Topological Accuracy** (avg-path-length error
on real routes). GeoTIFF masks auto-read their bounds.

```bash
python src/validate_real.py road_mask.tif                     # real model output
python src/validate_real.py --demo --place "Indiranagar, Bengaluru"   # dry-run today
```

Run it on an urban *and* a rural/forested tile to evidence generalisation across
terrains.

## Architecture

```
titanarch/
  src/
    roadgraph.py     # analysis engine: criticality, efficiency, resilience, ablation, repair, geojson
    osm.py           # fetch + simplify a real city's roads from OpenStreetMap
    mask_to_graph.py # road mask -> skeleton -> graph -> GAP HEALING (the novelty)
    synth.py         # reproducible synthetic occluded mask for the demo
  app/
    server.py        # Flask: /api/graph, /api/load_osm, /api/ablate, /api/recommend_repairs, /api/load_mask
    static/          # Leaflet dashboard (index.html, app.js, styles.css)
  colab/
    road_segmentation.ipynb  # GPU: fine-tune a road segmenter, export a mask
```

Design choice: we use **networkx** for the well-solved graph math and keep our
own code for the parts that are the product — the resilience stress-test and
Resilience Index, the interactive ablation, and (next) the gap-healing that
turns a broken road mask into a routable network.

---

## Run it

```powershell
# from the titanarch folder
& "<python>" app\server.py --port 8780
```

Then open <http://127.0.0.1:8780>, type e.g. `Indiranagar, Bengaluru`, click
**Load**, and start closing junctions.

Requires: `networkx`, `requests`, `flask` (`pip install networkx requests flask`).

---

## Notes on data

- Road network: **OpenStreetMap** via the public Overpass API — free, global,
  no key. Geocoding via Nominatim.
- Everything the tool reports is computed from the actual fetched network; the
  numbers change with the place and radius you load.
