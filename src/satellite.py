"""
Titan Arch  --  satellite.py

Fetches real satellite imagery for an area, with no API key, by stitching
Esri "World Imagery" web-map tiles. Returns the RGB image AND its exact
geographic bounding box, so the downstream road mask is georeferenced and
drops straight into Titan Arch.

This removes the two friction points of the GPU step: you no longer have to
find a satellite image or type its bounding box by hand -- give a place/bbox
and you get analysis-ready imagery.

Pure HTTP + numpy/PIL (no GPU). Used by the Colab segmentation notebook and
testable locally.
"""

from __future__ import annotations

import io
import math

import numpy as np
import requests
from PIL import Image

ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
UA = {"User-Agent": "TitanArch/0.1 (anuragp2003b@gmail.com)"}
TILE = 256


# --------------------------------------------------------------------------- #
#  slippy-map tile math (Web Mercator)
# --------------------------------------------------------------------------- #
def _lonlat_to_tilexy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1 - math.asinh(math.tan(lat_r)) / math.pi) / 2 * n
    return x, y


def _tilexy_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


# --------------------------------------------------------------------------- #
#  fetch + stitch
# --------------------------------------------------------------------------- #
def fetch_imagery(bbox, zoom=18, max_tiles=64, session=None):
    """
    bbox = (south, west, north, east). Returns (rgb_uint8_HxWx3, exact_bbox).
    The returned bbox is the true extent of the stitched, tile-aligned image
    (slightly larger than requested) -- use IT for georeferencing.
    """
    s, w, n, e = bbox
    sess = session or requests.Session()

    # tile index ranges covering the bbox (note: y grows southward)
    x0f, y0f = _lonlat_to_tilexy(n, w, zoom)   # top-left
    x1f, y1f = _lonlat_to_tilexy(s, e, zoom)   # bottom-right
    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))
    nx, ny = (x1 - x0 + 1), (y1 - y0 + 1)
    if nx * ny > max_tiles:
        raise ValueError(f"area needs {nx*ny} tiles (>{max_tiles}); "
                         f"lower the zoom or shrink the area")

    canvas = Image.new("RGB", (nx * TILE, ny * TILE))
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            url = ESRI.format(z=zoom, x=ix, y=iy)
            r = sess.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            canvas.paste(tile, ((ix - x0) * TILE, (iy - y0) * TILE))

    # exact extent of the stitched (tile-aligned) canvas
    north, west = _tilexy_to_lonlat(x0, y0, zoom)
    south, east = _tilexy_to_lonlat(x1 + 1, y1 + 1, zoom)
    return np.asarray(canvas), (south, west, north, east)


def geocode_bbox(place, half_m=1100):
    """Place name -> a square bbox around it (Nominatim)."""
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": place, "format": "json", "limit": 1},
                     headers=UA, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError(f"could not geocode '{place}'")
    lat, lon = float(js[0]["lat"]), float(js[0]["lon"])
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon), (lat, lon)


if __name__ == "__main__":
    import sys
    place = sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bengaluru"
    z = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    bbox, center = geocode_bbox(place, half_m=700)
    img, exact = fetch_imagery(bbox, zoom=z)
    print(f"{place}: image {img.shape}, mean RGB {img.reshape(-1,3).mean(0).round(1)}")
    print(f"requested bbox {tuple(round(x,5) for x in bbox)}")
    print(f"exact bbox     {tuple(round(x,5) for x in exact)}")
