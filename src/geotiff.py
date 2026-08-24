"""
Titan Arch  --  geotiff.py

Reads georeferenced rasters (GeoTIFF) so Titan Arch can ingest the indigenous
satellite products the brief centres on -- Resourcesat LISS-IV, Cartosat-3,
Sentinel-2 -- which ship as GeoTIFFs carrying both the pixels AND their map
coordinates. We pull the array and the exact WGS84 bounding box automatically,
so there is no manual coordinate entry.

Uses rasterio / GDAL (the commended geospatial stack).
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds


def _bbox_wgs84(ds):
    """Dataset bounds -> (south, west, north, east) in lat/lon."""
    left, bottom, right, top = ds.bounds
    if ds.crs and ds.crs.to_epsg() != 4326:
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326",
                                                     left, bottom, right, top)
    else:
        west, south, east, north = left, bottom, right, top
    return (south, west, north, east)


def _to_mask(ds):
    """First band as a uint8 mask (handles float/multi-band rasters)."""
    arr = ds.read(1).astype("float32")
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, 0, arr)
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return arr.astype("uint8")


def read_geotiff_path(path):
    with rasterio.open(path) as ds:
        return _to_mask(ds), _bbox_wgs84(ds)


def read_geotiff_bytes(raw):
    with MemoryFile(raw) as mf, mf.open() as ds:
        return _to_mask(ds), _bbox_wgs84(ds)


def is_geotiff(name="", raw=None):
    if name.lower().endswith((".tif", ".tiff")):
        return True
    if raw and len(raw) > 4 and raw[:2] in (b"II", b"MM"):   # TIFF magic
        return True
    return False


def write_geotiff(path, mask, bbox):
    """Write a mask as a georeferenced GeoTIFF (WGS84) -- handy for exporting an
    analysis-ready, geocoded road mask from the Colab notebook."""
    from rasterio.transform import from_bounds
    s, w, n, e = bbox
    h, wid = mask.shape[:2]
    tr = from_bounds(w, s, e, n, wid, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=wid, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=tr) as ds:
        ds.write(mask.astype("uint8"), 1)
