"""Local HTTP server for the 3dgs_io Cesium viewer.

Serves the viewer HTML at ``/`` and tile files from the specified directory
under ``/tiles/``.  All responses include permissive CORS headers so that
CesiumJS can fetch tiles without issues.
"""

from __future__ import annotations

import mimetypes
import posixpath
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import ClassVar

_VIEWER_DIR = Path(__file__).resolve().parent

# Ensure common 3D-tile MIME types are registered.
_EXTRA_TYPES: dict[str, str] = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".json": "application/json",
    ".b3dm": "application/octet-stream",
    ".i3dm": "application/octet-stream",
    ".pnts": "application/octet-stream",
    ".cmpt": "application/octet-stream",
    ".spz": "application/octet-stream",
}
for _ext, _mime in _EXTRA_TYPES.items():
    mimetypes.add_type(_mime, _ext)


class _ViewerHandler(SimpleHTTPRequestHandler):
    """Serves viewer assets from the package and tile files from a user directory."""

    tiles_directory: ClassVar[Path]  # set by ``create_server``

    # Silence per-request log lines (still logged on errors).
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        clean = posixpath.normpath(urllib.parse.unquote(parsed.path))

        # Serve viewer HTML at root.
        if clean in ("/", "/index.html"):
            self._serve_file(_VIEWER_DIR / "index.html")
            return

        # Serve tile files from the user-supplied directory.
        if clean.startswith("/tiles/"):
            relative = clean[len("/tiles/") :]
            target = (self.tiles_directory / relative).resolve()
            # Prevent path traversal.
            if not str(target).startswith(str(self.tiles_directory.resolve())):
                self.send_error(403, "Forbidden")
                return
            self._serve_file(target)
            return

        self.send_error(404, "Not Found")

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, "Not Found")
            return
        content_type, _ = mimetypes.guess_type(str(path))
        if content_type is None:
            content_type = "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def create_server(
    tiles_directory: Path,
    host: str = "localhost",
    port: int = 8080,
) -> HTTPServer:
    """Create an :class:`~http.server.HTTPServer` ready to ``.serve_forever()``.

    Parameters
    ----------
    tiles_directory:
        Directory containing the tileset (``tileset.json`` and its content
        files).  Served under ``/tiles/``.
    host:
        Bind address.  Defaults to ``"localhost"``.
    port:
        Bind port.  Defaults to ``8080``.
    """
    tiles_directory = Path(tiles_directory).resolve()

    # Create a handler subclass that captures the tiles directory so each
    # request instance can access it via ``self.tiles_directory``.
    handler = type(
        "_BoundHandler",
        (_ViewerHandler,),
        {"tiles_directory": tiles_directory},
    )

    server = HTTPServer((host, port), handler)
    return server
