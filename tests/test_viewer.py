"""Tests for the viewer module."""

from __future__ import annotations

import importlib
import json
import socket
import threading
import time
import urllib.request
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


class TestViewerServer:
    def test_serves_index_html(self, tileset_dir: Path) -> None:
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/")
            body = resp.read().decode()
            assert "3dgs_io Viewer" in body
            assert resp.status == 200
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
        finally:
            server.shutdown()

    def test_serves_tileset_json(self, tileset_dir: Path) -> None:
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/tiles/tileset.json")
            data = json.loads(resp.read())
            assert data["asset"]["version"] == "1.1"
            assert "application/json" in resp.headers["Content-Type"]
        finally:
            server.shutdown()

    def test_404_for_missing_file(self, tileset_dir: Path) -> None:
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(urllib.error.HTTPError, match="404"):
                urllib.request.urlopen(f"http://localhost:{port}/tiles/nonexistent.json")
        finally:
            server.shutdown()

    def test_path_traversal_blocked(self, tileset_dir: Path) -> None:
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://localhost:{port}/tiles/../../../etc/passwd")
        finally:
            server.shutdown()

    def test_cors_on_options(self, tileset_dir: Path) -> None:
        port = _free_port()
        server = create_server(tileset_dir, port=port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/tiles/tileset.json",
                method="OPTIONS",
            )
            resp = urllib.request.urlopen(req)
            assert resp.status == 200
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
        finally:
            server.shutdown()


class TestLaunchViewer:
    def test_raises_for_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            launch_viewer("/nonexistent/path", open_browser=False)

    def test_accepts_tileset_json_path(self, tileset_dir: Path) -> None:
        """launch_viewer should accept a direct path to tileset.json."""
        port = _free_port()

        def _run() -> None:
            launch_viewer(
                tileset_dir / "tileset.json",
                port=port,
                open_browser=False,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(0.5)  # let server start

        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/tiles/tileset.json")
            assert resp.status == 200
        finally:
            # Server will be cleaned up when thread is daemonized
            pass
