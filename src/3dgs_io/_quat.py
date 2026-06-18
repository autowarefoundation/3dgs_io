"""Internal quaternion <-> rotation-matrix helpers shared by cameras + rig_trajectories.

Both modules need to extract an xyzw quaternion from a 3x3 rotation matrix
(via Shepperd's method) and to rebuild a 3x3 from an xyzw quaternion. Keeping
the math in one place avoids the silent-drift hazard of two near-identical
Shepperd implementations.
"""

from __future__ import annotations

import numpy as np


def quat_from_rotation_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    """Extract an xyzw unit quaternion from a 3×3 rotation matrix.

    Uses Shepperd's branch selection (pick the diagonal element that
    maximises the divisor so the square root stays large) for numerical
    stability. The input is assumed to be approximately orthonormal —
    scale/shear in ``r`` will be silently absorbed into the resulting
    quaternion.
    """
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
