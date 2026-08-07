#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#   "numpy==2.0.2",
#   "scipy==1.13.1",
#   "scikit-fem==11.0.0",
# ]
# ///
"""Focused convention, analytical-anchor and FEM smoke tests."""

import csv
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inner_stokes_generator", ROOT / "generate.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load generate.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class ConventionTests(unittest.TestCase):
    def test_single_phase_mobility(self):
        for theta in (0.4, 1.0, math.pi / 2.0, 2.4):
            expected = -2.0 * math.sin(theta) ** 3 / (
                3.0 * (theta - math.sin(theta) * math.cos(theta))
            )
            self.assertAlmostEqual(GENERATOR.cox_mobility(theta, 0.0), expected, places=13)

    def test_right_angle_branch_is_phase_exchange_symmetric(self):
        for ratio in (0.02, 0.1, 0.5, 2.0, 10.0, 50.0):
            self.assertAlmostEqual(
                GENERATOR.corrected_right_angle_Q(ratio),
                GENERATOR.corrected_right_angle_Q(1.0 / ratio),
                places=14,
            )

    def test_exact_single_phase_right_angle(self):
        expected = 1.0 + GENERATOR.EULER_GAMMA - math.log(2.0)
        self.assertAlmostEqual(GENERATOR.scott_hocking_right_angle_Q(), expected, places=15)

    def test_huh_scriven_outer_solution_satisfies_interface_conditions(self):
        theta = 1.1
        ratio = 3.0
        a, b = GENERATOR.huh_scriven_coefficients(theta, ratio)
        self.assertIsNotNone(b)
        av, af, ass = GENERATOR._streamfunction_rows(theta)
        bv, bf, bss = GENERATOR._streamfunction_rows(theta)
        self.assertAlmostEqual(float(av @ a), 0.0, places=12)
        self.assertAlmostEqual(float(bv @ b), 0.0, places=12)
        self.assertAlmostEqual(float(af @ a - bf @ b), 0.0, places=12)
        self.assertAlmostEqual(float(ass @ a - ratio * bss @ b), 0.0, places=12)

    def test_scott_table_convention_and_derived_columns(self):
        with (ROOT / "data" / "scott-hocking-m0.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        for row in rows:
            theta = float(row["theta_rad"])
            qi = float(row["Qi_scott"])
            q = float(row["Q_chan"])
            log_c = float(row["log_c"])
            self.assertAlmostEqual(q, 1.0 + qi, places=12)
            self.assertAlmostEqual(log_c, 1.0 + math.log(math.sin(theta)) - q, places=10)
            self.assertAlmostEqual(float(row["c"]), math.exp(log_c), places=9)

    def test_schema_identifies_Q_as_the_runtime_quantity(self):
        schema = json.loads((ROOT / "table.schema.json").read_text())
        self.assertIn("Q", schema["required"])
        self.assertEqual(schema["properties"]["converged"]["const"], True)


class FiniteElementSmokeTest(unittest.TestCase):
    def test_one_phase_force_has_the_published_sign_and_scale(self):
        result = GENERATOR.solve_wedge(
            math.pi / 2.0,
            0.0,
            GENERATOR.MeshSpec(
                inner_radius=1.0e-2,
                outer_radius=1.0e2,
                radial_cells=32,
            ),
            GENERATOR.FitWindow(4.0, 25.0),
        )
        self.assertGreater(result.slope_fitted, 0.0)
        self.assertLess(result.slope_relative_error, 0.10)
        self.assertLess(
            abs(result.Q - GENERATOR.scott_hocking_right_angle_Q()), 0.06
        )
        self.assertLess(result.linear_system_backward_error, 1.0e-10)
        self.assertLess(result.constraint_relative_error, 1.0e-10)

    def test_two_phase_force_respects_phase_exchange(self):
        mesh = GENERATOR.MeshSpec(
            inner_radius=1.0e-2,
            outer_radius=1.0e2,
            radial_cells=32,
        )
        fit = GENERATOR.FitWindow(4.0, 25.0)
        direct = GENERATOR.solve_wedge(math.pi / 3.0, 0.2, mesh, fit)
        exchanged = GENERATOR.solve_wedge(2.0 * math.pi / 3.0, 5.0, mesh, fit)
        self.assertAlmostEqual(direct.Q, exchanged.Q, places=11)
        self.assertLess(direct.linear_system_backward_error, 1.0e-10)
        self.assertLess(exchanged.linear_system_backward_error, 1.0e-10)
        self.assertLess(direct.constraint_relative_error, 1.0e-10)
        self.assertLess(exchanged.constraint_relative_error, 1.0e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
