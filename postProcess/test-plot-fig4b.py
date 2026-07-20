#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Regression tests for the Figure 4b external-validation grid."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("plot-fig4b.py")
SPEC = importlib.util.spec_from_file_location("plot_fig4b", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class UniformHeightGateTests(unittest.TestCase):
    def test_bezier_density_cannot_hide_height_weighted_error(self) -> None:
        parameter = np.linspace(0.0, 1.0, 421)
        reference_height = parameter**3
        reference_ca = np.zeros_like(reference_height)
        branch_height = np.linspace(0.0, 1.0, 421)
        branch_ca = 2.0e-4 * branch_height

        raw_errors = np.interp(
            reference_height, branch_height, branch_ca
        ) - reference_ca
        raw_rms = float(np.sqrt(np.mean(raw_errors**2)))
        _, uniform_rms = PLOT.uniform_height_error_metrics(
            branch_height,
            branch_ca,
            reference_height,
            reference_ca,
            421,
        )

        self.assertLess(raw_rms, 1.0e-4)
        self.assertGreater(uniform_rms, 1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
