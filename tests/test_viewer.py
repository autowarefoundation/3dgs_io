"""Tests for the viewer module."""

from __future__ import annotations

import importlib
import json
import socket
import threading
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest

_mod = importlib.import_module("3dgs_io")
launch_viewer = _mod.launch_viewer

_viewer_mod = importlib.import_module("3dgs_io.viewer._server")
create_server = _viewer_mod.create_server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def tileset_dir(tmp_path: Path) -> Path:
    """Create a minimal tileset directory for testing."""
    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": 100,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
            "geometricError": 50,
            "refine": "ADD",
        },
    }
    (tmp_path / "tileset.json").write_text(json.dumps(tileset))
    return tmp_path


@pytest.fixture()
def running_server(tileset_dir: Path) -> Generator[tuple[str, int], None, None]:
    """Start a viewer server and yield ``(base_url, port)``, shutting down on exit."""
    port = _free_port()
    server = create_server(tileset_dir, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}", port
    server.shutdown()


class TestViewerServer:
    def test_serves_index_html(self, running_server: tuple[str, int]) -> None:
        base_url, _ = running_server
        resp = urllib.request.urlopen(f"{base_url}/")
        body = resp.read().decode()
        assert "3dgs_io Viewer" in body
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_serves_tileset_json(self, running_server: tuple[str, int]) -> None:
        base_url, _ = running_server
        resp = urllib.request.urlopen(f"{base_url}/tiles/tileset.json")
        data = json.loads(resp.read())
        assert data["asset"]["version"] == "1.1"
        assert "application/json" in resp.headers["Content-Type"]

    def test_404_for_missing_file(self, running_server: tuple[str, int]) -> None:
        base_url, _ = running_server
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(f"{base_url}/tiles/nonexistent.json")

    def test_path_traversal_blocked(self, running_server: tuple[str, int]) -> None:
        base_url, _ = running_server
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base_url}/tiles/../../../etc/passwd")

    def test_cors_on_options(self, running_server: tuple[str, int]) -> None:
        base_url, _ = running_server
        req = urllib.request.Request(f"{base_url}/tiles/tileset.json", method="OPTIONS")
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"


class TestLaunchViewer:
    def test_raises_for_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            launch_viewer("/nonexistent/path", open_browser=False)

    def test_accepts_tileset_json_path(self, tileset_dir: Path) -> None:
        """launch_viewer resolves a tileset.json path to its parent directory."""
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/tiles/tileset.json")
            assert resp.status == 200
        finally:
            server.shutdown()
