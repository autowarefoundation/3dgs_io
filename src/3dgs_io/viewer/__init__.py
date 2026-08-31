"""Cesium-based 3D Tiles viewer for Gaussian Splatting assets.

Usage::

    from 3dgs_io.viewer import launch_viewer

    # Opens browser and serves tiles from the given directory.
    launch_viewer("output/tiles")
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from ._server import create_server

__all__ = ["launch_viewer"]


def launch_viewer(
    tiles_directory: str | Path,
    *,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
    tileset_name: str = "tileset.json",
) -> None:
    """Launch a local Cesium viewer for a 3D Tiles directory.

    The function starts an HTTP server that serves the viewer page and the
    tile files, then blocks until interrupted (Ctrl-C).

    Parameters
    ----------
    tiles_directory:
        Path to the directory that contains the tileset.  If a direct path to
        ``tileset.json`` is given, its parent directory is used.
    host:
        Bind address.  ``"localhost"`` (default) restricts access to the local
        machine; use ``"0.0.0.0"`` to allow access from the network.
    port:
        HTTP port.  Defaults to ``8080``.
    open_browser:
        Whether to open the system browser automatically.
    tileset_name:
        Name of the tileset JSON file inside the directory.  The viewer URL
        is constructed so that CesiumJS loads ``/tiles/<tileset_name>``
        automatically.
    """
    tiles_dir = Path(tiles_directory)
    if tiles_dir.is_file():
        tileset_name = tiles_dir.name
        tiles_dir = tiles_dir.parent

    if not tiles_dir.is_dir():
        msg = f"Tiles directory does not exist: {tiles_dir}"
        raise FileNotFoundError(msg)

    server = create_server(tiles_dir, host=host, port=port)

    url = f"http://{host}:{port}/?tileset=tiles/{tileset_name}"
    print(f"Serving viewer at {url}")
    print(f"Tiles directory: {tiles_dir}")
    print("Press Ctrl-C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
