#!/usr/bin/env python3
"""Audit the frozen Scott--Hocking reference with the Python standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unittest
from pathlib import Path
from typing import Sequence


GENERATOR = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = GENERATOR / "scott_hocking.py"
NODES = GENERATOR / "data" / "scott-hocking-m0-nodes.csv"
CHECKPOINTS = GENERATOR / "data" / "scott-hocking-m0-checkpoints.csv"
MANIFEST = GENERATOR / "data" / "scott-hocking-m0.manifest.json"
PUBLISHED = GENERATOR / "data" / "scott-hocking-m0.csv"
HEADER = REPOSITORY / "src-local" / "gle-slip-scott-data.h"
EULER_GAMMA = 0.57721566490153286061
RIGHT_ANGLE_QI = EULER_GAMMA - math.log(2.0)
LARGE_ANGLE_CONSTANT = RIGHT_ANGLE_QI - 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if d0 == 0.0 or (slope > 0.0) != (d0 > 0.0):
        return 0.0
    if (d0 > 0.0) != (d1 > 0.0) and abs(slope) > 3.0 * abs(d0):
        return 3.0 * d0
    return slope


def pchip(
    angles: Sequence[float],
    values: Sequence[float],
    query: float,
    *,
    small_angle: bool = False,
) -> float:
    widths = [right - left for left, right in zip(angles[:-1], angles[1:])]
    secants = [
        (right - left) / width
        for left, right, width in zip(values[:-1], values[1:], widths)
    ]
    tangents = [0.0] * len(angles)
    tangents[0] = endpoint_slope(widths[0], widths[1], secants[0], secants[1])
    for index in range(1, len(angles) - 1):
        left, right = secants[index - 1], secants[index]
        if left != 0.0 and right != 0.0 and (left > 0.0) == (right > 0.0):
            weight1 = 2.0 * widths[index] + widths[index - 1]
            weight2 = widths[index] + 2.0 * widths[index - 1]
            tangents[index] = (weight1 + weight2) / (
                weight1 / left + weight2 / right
            )
    tangents[-1] = endpoint_slope(
        widths[-1], widths[-2], secants[-1], secants[-2]
    )
    if small_angle:
        tangents[0] = 0.0
        tangents[1] = 3.0 * values[1] / widths[0] - 0.156 * widths[0]
    interval = 0
    while interval < len(angles) - 2 and query > angles[interval + 1]:
        interval += 1
    local = (query - angles[interval]) / widths[interval]
    local2, local3 = local * local, local * local * local
    return (
        (2.0 * local3 - 3.0 * local2 + 1.0) * values[interval]
        + (local3 - 2.0 * local2 + local)
        * widths[interval]
        * tangents[interval]
        + (-2.0 * local3 + 3.0 * local2) * values[interval + 1]
        + (local3 - local2) * widths[interval] * tangents[interval + 1]
    )


def endpoint_augmented(
    angles: Sequence[float], qi: Sequence[float]
) -> tuple[list[float], list[float], list[float], list[float]]:
    lower_a, lower_r = [0.0], [0.0]
    upper_a, upper_r = [0.5 * math.pi], [RIGHT_ANGLE_QI - 2.0]
    for theta, value in zip(angles, qi):
        if theta < 0.5 * math.pi:
            lower_a.append(theta)
            lower_r.append(value - math.log(theta / 3.0))
        elif theta > 0.5 * math.pi:
            upper_a.append(theta)
            upper_r.append(value - math.pi / (math.pi - theta))
    lower_a.append(0.5 * math.pi)
    lower_r.append(RIGHT_ANGLE_QI - math.log((0.5 * math.pi) / 3.0))
    upper_a.append(math.pi)
    upper_r.append(LARGE_ANGLE_CONSTANT)
    return lower_a, lower_r, upper_a, upper_r


def interpolate(angles: Sequence[float], qi: Sequence[float], theta: float) -> float:
    lower_a, lower_r, upper_a, upper_r = endpoint_augmented(angles, qi)
    if theta <= 0.5 * math.pi:
        return math.log(theta / 3.0) + pchip(
            lower_a, lower_r, theta, small_angle=True
        )
    return math.pi / (math.pi - theta) + pchip(upper_a, upper_r, theta)


def header_array(source: str, name: str) -> list[float]:
    match = re.search(
        rf"static const double {name}\[\d+\]\s*=\s*\{{(.*?)\}};",
        source,
        re.DOTALL,
    )
    if not match:
        raise AssertionError(f"missing generated array {name}")
    return [float(value.strip()) for value in match.group(1).split(",")]


class ScottReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with NODES.open(newline="", encoding="utf-8") as stream:
            cls.nodes = list(csv.DictReader(stream))
        with CHECKPOINTS.open(newline="", encoding="utf-8") as stream:
            cls.checkpoints = list(csv.DictReader(stream))
        cls.angles = [float(row["theta_rad"]) for row in cls.nodes]
        cls.qi = [float(row["Qi"]) for row in cls.nodes]

    def test_manifest_hashes_and_convergence(self) -> None:
        self.assertTrue(self.manifest["converged"])
        self.assertEqual(self.manifest["generator_sha256"], sha256(SOURCE))
        self.assertEqual(self.manifest["node_csv_sha256"], sha256(NODES))
        self.assertEqual(self.manifest["checkpoint_csv_sha256"], sha256(CHECKPOINTS))
        self.assertEqual(self.manifest["node_count"], len(self.nodes))
        self.assertEqual(self.manifest["checkpoint_count"], len(self.checkpoints))
        self.assertLessEqual(self.manifest["max_node_sensitivity_Qi"], 8.0e-5)
        self.assertLessEqual(
            self.manifest["max_midpoint_interpolation_error_Qi"], 8.0e-5
        )
        self.assertLessEqual(
            self.manifest["max_midpoint_error_budget_Qi"], 2.5e-4
        )
        self.assertEqual(len(self.checkpoints), len(self.nodes) + 1)
        self.assertAlmostEqual(
            float(self.checkpoints[0]["theta_rad"]),
            0.5 * self.angles[0],
            delta=5.0e-16,
        )
        self.assertAlmostEqual(
            float(self.checkpoints[-1]["theta_rad"]),
            0.5 * (self.angles[-1] + math.pi),
            delta=5.0e-16,
        )

    def test_generated_header_matches_nodes(self) -> None:
        source = HEADER.read_text(encoding="utf-8")
        self.assertIn(self.manifest["generator_sha256"], source)
        self.assertIn(self.manifest["node_csv_sha256"], source)
        self.assertIn(self.manifest["checkpoint_csv_sha256"], source)
        expected = endpoint_augmented(self.angles, self.qi)
        names = (
            "gle_scott_lower_theta",
            "gle_scott_lower_regular",
            "gle_scott_upper_theta",
            "gle_scott_upper_regular",
        )
        for name, values in zip(names, expected):
            self.assertEqual(header_array(source, name), values)

    def test_all_independent_midpoints(self) -> None:
        maximum = 0.0
        for row in self.checkpoints:
            theta = float(row["theta_rad"])
            actual = interpolate(self.angles, self.qi, theta)
            recorded = float(row["Qi_interpolated"])
            self.assertAlmostEqual(actual, recorded, delta=5.0e-14)
            direct = float(row["Qi_direct"])
            discrepancy = abs(actual - direct)
            self.assertAlmostEqual(
                discrepancy,
                float(row["abs_interpolation_error_Qi"]),
                delta=5.0e-14,
            )
            maximum = max(maximum, discrepancy)
        self.assertAlmostEqual(
            maximum,
            self.manifest["max_midpoint_interpolation_error_Qi"],
            delta=5.0e-14,
        )

    def test_published_scott_table_and_chan_columns(self) -> None:
        errors = []
        with PUBLISHED.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                theta = float(row["theta_rad"])
                published = float(row["Qi_scott"])
                errors.append(abs(interpolate(self.angles, self.qi, theta) - published))
        self.assertAlmostEqual(
            max(errors), self.manifest["max_published_table_error_Qi"], delta=5.0e-14
        )
        for row in self.nodes:
            theta = float(row["theta_rad"])
            value = float(row["Qi"])
            self.assertAlmostEqual(float(row["Q_chan"]), 1.0 + value, delta=5.0e-14)
            self.assertAlmostEqual(
                float(row["log_c"]), math.log(math.sin(theta)) - value, delta=5.0e-14
            )

    def test_only_right_angle_is_analytic_exact(self) -> None:
        self.assertAlmostEqual(
            interpolate(self.angles, self.qi, 0.5 * math.pi),
            RIGHT_ANGLE_QI,
            delta=5.0e-15,
        )
        public_headers = (
            (REPOSITORY / "src-local" / "gle-model.h").read_text(encoding="utf-8")
            + (REPOSITORY / "src-local" / "gle-slip-reference.h").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("SINGLE_PHASE_EXACT", public_headers)
        self.assertNotIn("single_phase_exact", public_headers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
