"""Run the splatsim USDZ → scene-bundle converter via ``python -m 3dgs_io``."""

from __future__ import annotations

from .scene_bundle_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
