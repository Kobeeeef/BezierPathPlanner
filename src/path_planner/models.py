from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BezierSegment:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "P0": {"x": float(self.p0[0]), "y": float(self.p0[1])},
            "P1": {"x": float(self.p1[0]), "y": float(self.p1[1])},
            "P2": {"x": float(self.p2[0]), "y": float(self.p2[1])},
            "P3": {"x": float(self.p3[0]), "y": float(self.p3[1])},
        }
