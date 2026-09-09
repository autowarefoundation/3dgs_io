"""Tests for the WGS84 helpers and the anchor↔lanelet2 cross-check."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_geodesy = importlib.import_module("3dgs_io._geodesy")


def test_geodetic_ecef_roundtrip() -> None:
    for lat, lon, h in [(35.6216, 139.7779, 9.9), (-33.9, 18.4, 120.0), (0.0, 0.0, 0.0)]:
        ecef = _geodesy.geodetic_to_ecef(lat, lon, h).reshape(3)
        lat2, lon2, h2 = _geodesy.ecef_to_geodetic(ecef)
        assert abs(lat2 - lat) < 1e-9
        assert abs(lon2 - lon) < 1e-9
        assert abs(h2 - h) < 1e-4


def test_known_ecef_point() -> None:
    # Greenwich equator sea level: (a, 0, 0).
    ecef = _geodesy.geodetic_to_ecef(0.0, 0.0, 0.0).reshape(3)
    np.testing.assert_allclose(ecef, [6378137.0, 0.0, 0.0], atol=1e-6)


def test_enu_rotation_is_proper() -> None:
    r = _geodesy.enu_to_ecef_rotation(35.6, 139.78)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    # The ENU up axis points away from the Earth centre.
    up_ecef = r @ np.array([0.0, 0.0, 1.0])
    centre_dir = _geodesy.geodetic_to_ecef(35.6, 139.78, 0.0).reshape(3)
    assert np.dot(up_ecef, centre_dir / np.linalg.norm(centre_dir)) > 0.99


def _anchor_at(lat: float, lon: float, h: float = 0.0) -> np.ndarray:
    anchor = np.eye(4)
    anchor[:3, :3] = _geodesy.enu_to_ecef_rotation(lat, lon)
    anchor[:3, 3] = _geodesy.geodetic_to_ecef(lat, lon, h).reshape(3)
    return anchor


def _osm(lat: float, lon: float) -> bytes:
    return f'<osm><node id="1" lat="{lat}" lon="{lon}"/></osm>'.encode()


def test_validate_anchor_accepts_nearby_map() -> None:
    _geodesy.validate_anchor_against_lanelet2(_anchor_at(35.62, 139.78), _osm(35.63, 139.79))


def test_validate_anchor_rejects_distant_map() -> None:
    # The incident: anchor at an MGRS grid corner ~95 km from the map.
    with pytest.raises(ValueError, match="km apart"):
        _geodesy.validate_anchor_against_lanelet2(
            _anchor_at(35.2253, 138.8049, 748.0), _osm(35.6216, 139.7779)
        )


def test_validate_anchor_skips_unusable_map() -> None:
    _geodesy.validate_anchor_against_lanelet2(_anchor_at(35.62, 139.78), b"<osm/>")
    _geodesy.validate_anchor_against_lanelet2(_anchor_at(35.62, 139.78), b"not xml")
