#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#   "numpy==2.0.2",
#   "scipy==1.13.1",
# ]
# ///
"""Small independent numerical reruns of the Scott--Hocking generator."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scott_hocking_generator", ROOT / "scott_hocking.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load Scott--Hocking generator")
SCOTT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCOTT
SPEC.loader.exec_module(SCOTT)


class ScottSolverTests(unittest.TestCase):
    def test_published_and_analytic_anchors(self) -> None:
        published = SCOTT.solve_qi(0.1, spacing=0.1, radius=40.0)
        self.assertAlmostEqual(published.qi, -3.399640, delta=3.0e-5)
        right = SCOTT.solve_qi(0.5 * math.pi, spacing=0.1, radius=40.0)
        self.assertAlmostEqual(right.qi, SCOTT.RIGHT_ANGLE_QI, delta=2.0e-6)
        self.assertLess(max(published.relative_residual, right.relative_residual), 2.0e-13)

    def test_discretisation_refinement(self) -> None:
        coarse = SCOTT.solve_qi(2.95, spacing=0.1, radius=60.0)
        fine = SCOTT.solve_qi(2.95, spacing=0.05, radius=60.0)
        self.assertLess(abs(fine.qi - coarse.qi), 3.0e-6)

    def test_asymptotic_endpoint_cells_have_direct_checks(self) -> None:
        with SCOTT.DEFAULT_TABLE.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        angles = [float(row["theta_rad"]) for row in rows]
        qi = [float(row["Qi"]) for row in rows]
        theta = 0.5 * (angles[-1] + math.pi)
        direct = SCOTT.solve_qi(theta, spacing=0.1, radius=240.0)
        interpolated = SCOTT.interpolate_qi(angles, qi, theta)
        self.assertLess(abs(interpolated - direct.qi), 8.0e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
