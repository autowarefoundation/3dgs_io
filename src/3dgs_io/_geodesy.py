"""Minimal WGS84 geodesy helpers (no external projection dependency).

Used to sanity-check that a scene bundle's ``ecef_anchor`` actually sits at
the location described by the embedded Lanelet2 ``map.osm``. A mis-derived
anchor (e.g. one that points at an MGRS 100 km grid corner instead of the
scene) renders fine standalone but silently breaks every geodetic consumer
(CARLA bridges, Autoware map alignment), so the writer refuses to produce
such a bundle.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np

_log = logging.getLogger(__name__)

__all__ = [
    "ecef_to_geodetic",
    "enu_to_ecef_rotation",
    "geodetic_to_ecef",
    "lanelet2_mean_ecef",
    "validate_anchor_against_lanelet2",
]

# WGS84 ellipsoid.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def geodetic_to_ecef(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    height_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """WGS84 geodetic (degrees, ellipsoidal metres) → ECEF ``(..., 3)`` metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(height_m, dtype=np.float64)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[float, float, float]:
    """ECEF metres ``(3,)`` → WGS84 ``(lat_deg, lon_deg, height_m)``.

    Fixed-point iteration on the latitude; converges to sub-millimetre in a
    handful of rounds for any point near the Earth's surface.
    """
    x, y, z = (float(v) for v in np.asarray(ecef, dtype=np.float64).reshape(3))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    sin_lat = np.sin(lat)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    h = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation matrix (3×3) from the local ENU frame at ``lat/lon`` to ECEF."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ],
        dtype=np.float64,
    )


def lanelet2_mean_ecef(map_osm: bytes, *, max_nodes: int = 1000) -> np.ndarray | None:
    """Mean ECEF position of the first ``max_nodes`` Lanelet2 nodes, or ``None``.

    Node elevation (``ele`` tag) is deliberately ignored: it is orthometric
    (geoid-relative) while this check only needs tens-of-kilometres horizontal
    accuracy, far above the ~40 m geoid undulation.

    Returns ``None`` when the document is not parseable XML or carries no
    ``lat``/``lon`` node attributes, so callers can skip the check for
    placeholder files.
    """
    try:
        root = ET.fromstring(map_osm)  # noqa: S314 - trusted local map file
    except ET.ParseError:
        _log.warning("map.osm is not parseable XML; skipping anchor cross-check")
        return None
    lats: list[float] = []
    lons: list[float] = []
    for node in root.iter("node"):
        lat = node.get("lat")
        lon = node.get("lon")
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))
        if len(lats) >= max_nodes:
            break
    if not lats:
        return None
    return geodetic_to_ecef(np.mean(lats), np.mean(lons)).reshape(3)


def validate_anchor_against_lanelet2(
    ecef_anchor: np.ndarray,
    map_osm: bytes,
    *,
    max_offset_m: float = 10_000.0,
) -> None:
    """Reject an ``ecef_anchor`` that is far from the embedded Lanelet2 map.

    ``ecef_anchor`` is the row-major 4×4 world→ECEF transform; its translation
    column is the geodetic location of the scene's world origin, which must sit
    inside (or near) the area the bundled ``map.osm`` describes. Silently
    returns when the map carries no usable nodes.
    """
    map_ecef = lanelet2_mean_ecef(map_osm)
    if map_ecef is None:
        return
    anchor = np.asarray(ecef_anchor, dtype=np.float64).reshape(4, 4)
    anchor_t = anchor[:3, 3]
    offset = float(np.linalg.norm(anchor_t - map_ecef))
    if offset <= max_offset_m:
        return
    anchor_lat, anchor_lon, anchor_h = ecef_to_geodetic(anchor_t)
    map_lat, map_lon, _ = ecef_to_geodetic(map_ecef)
    raise ValueError(
        "ecef_anchor is inconsistent with the embedded Lanelet2 map: the world "
        f"origin decodes to lat={anchor_lat:.6f} lon={anchor_lon:.6f} "
        f"h={anchor_h:.1f} m but the map.osm nodes centre on "
        f"lat={map_lat:.6f} lon={map_lon:.6f} — {offset / 1000.0:.1f} km apart "
        f"(limit {max_offset_m / 1000.0:.1f} km). This usually means the anchor "
        "was derived at an MGRS grid origin or a recentring shift was applied "
        "to the geometry but not to the anchor."
    )
