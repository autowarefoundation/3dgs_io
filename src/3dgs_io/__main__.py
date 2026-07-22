"""Run the scene-USDZ packer via ``python -m 3dgs_io``."""

from __future__ import annotations

import sys

from .scene_usdz_cli import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export-tiles":
        from .usdz_tiles_export_cli import main as export_main

        raise SystemExit(export_main(sys.argv[2:]))
    raise SystemExit(main())
