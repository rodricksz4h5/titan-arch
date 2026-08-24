```
============================================================
TITAN ARCH — REAL-DATA VALIDATION REPORT CARD
footprint (S,W,N,E): (12.96521, 77.63217, 12.98138, 77.64876)
============================================================
PIXEL (extracted roads vs OpenStreetMap truth)
  IoU                  93.8%
  Dice                 96.8%
  Precision            99.7%
  Recall               94.1%
  F1                   96.8%
  Relaxed_IoU_3px      99.7%
------------------------------------------------------------
GRAPH / ROUTING
  nodes (pred / OSM)   12091 / 911
  components  47 -> 5  (MST/DSU bridges: 162)
  Connectivity Ratio   +4%  (LCC growth after healing)
  Topological Accuracy 11.4% APL error  over 150 routes
============================================================
```
