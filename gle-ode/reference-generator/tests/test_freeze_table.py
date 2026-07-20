#!/usr/bin/env python3
"""Unit tests for deterministic table freezing and interpolation."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_table", ROOT / "freeze_table.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load freeze_table.py")
FREEZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FREEZE
SPEC.loader.exec_module(FREEZE)


FIELDS = [
    "theta_deg",
    "theta_rad",
    "viscosity_ratio",
    "log10_viscosity_ratio",
    "Q",
    "converged",
    "estimated_error_Q",
    "slope_relative_error",
    "plateau_half_drift_Q",
    "max_linear_system_backward_error",
    "max_constraint_relative_error",
    "generator_version",
    "generator_source_sha256",
]

SOURCE_SHA = hashlib.sha256((ROOT / "generate.py").read_bytes()).hexdigest()


def row(theta, log_m):
    return {
        "theta_deg": theta,
        "theta_rad": theta * 3.141592653589793 / 180.0,
        "viscosity_ratio": 10.0**log_m,
        "log10_viscosity_ratio": log_m,
        "Q": 0.01 * theta + 0.2 * log_m,
        "converged": True,
        "estimated_error_Q": 1.0e-5,
        "slope_relative_error": 2.0e-5,
        "plateau_half_drift_Q": 3.0e-5,
        "max_linear_system_backward_error": 4.0e-14,
        "max_constraint_relative_error": 5.0e-14,
        "generator_version": "2",
        "generator_source_sha256": SOURCE_SHA,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path, _theta=None, **overrides):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    theta_grid = sorted({float(item["theta_deg"]) for item in rows})
    log_grid = sorted(
        {float(item["log10_viscosity_ratio"]) for item in rows}
    )
    manifest = {
        "generator_version": "2",
        "generator_source_sha256": SOURCE_SHA,
        "schema": "table.schema.json",
        "table": path.name,
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "interpolated_quantity": "Q",
        "method": "test",
        "all_converged": True,
        "theta_deg_grid": theta_grid,
        "log10_M_grid": log_grid,
        "mesh": {"inner_radius": 1e-8},
        "fit_window": {"lower": 1e5, "upper": 1e7},
        "tolerance_Q": 1e-3,
        "tolerance_slope": 2e-3,
        "tolerance_linear": 1e-10,
        "convergence_checks": ["test"],
        "references": [],
    }
    manifest.update(overrides)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class FreezeTableTests(unittest.TestCase):
    def test_merge_check_symmetry_and_emit(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shards = []
            for theta in (30.0, 150.0):
                path = work / f"theta-{theta:g}.csv"
                write_csv(path, [row(theta, -1.0), row(theta, 0.0)])
                write_manifest(path, theta)
                shards.append(path)

            table = work / "table.csv"
            self.assertEqual(
                FREEZE.merge_command(
                    Namespace(shards=[str(path) for path in shards], output=str(table))
                ),
                0,
            )

            checkpoints = work / "checkpoints.csv"
            direct = row(90.0, -0.5)
            symmetric = row(90.0, 0.5)
            symmetric["Q"] = direct["Q"]
            write_csv(checkpoints, [direct, symmetric])
            write_manifest(checkpoints, 90.0)
            audit = work / "audit.csv"
            self.assertEqual(
                FREEZE.check_command(
                    Namespace(
                        table=str(table),
                        checkpoints=str(checkpoints),
                        output=str(audit),
                        tolerance_observed_Q=1.0e-14,
                        tolerance_error_budget_Q=3.0e-5,
                        tolerance_right_angle_Q=1.0e-3,
                        tolerance_symmetry_Q=2.0,
                    )
                ),
                0,
            )

            header = work / "table-data.h"
            self.assertEqual(
                FREEZE.emit_c_command(
                    Namespace(table=str(table), output=str(header))
                ),
                0,
            )
            text = header.read_text(encoding="utf-8")
            self.assertIn("GLE_SLIP_TABLE_THETA_COUNT 2", text)
            self.assertIn("GLE_SLIP_TABLE_LOGM_COUNT 2", text)
            self.assertIn("GLE_SLIP_TABLE_THETA_STENCIL_NODES 4", text)
            self.assertIn("GLE_SLIP_TABLE_LOGM_STENCIL_NODES 5", text)
            self.assertIn("gle_slip_table_Q", text)
            self.assertIn("gle_slip_table_max_checkpoint_error_budget_Q", text)

            source = work / "compile-table.c"
            source.write_text(
                '#include "table-data.h"\n'
                "int main(void) {\n"
                "  return gle_slip_table_Q[0][0] == 0.0;\n"
                "}\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "cc",
                    "-std=c99",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    str(source),
                    "-o",
                    str(work / "compile-table"),
                ],
                cwd=work,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_merge_selects_exact_grid_from_nonrectangular_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            old = work / "old.csv"
            write_csv(
                old,
                [
                    row(theta, log_m)
                    for theta in (30.0, 60.0, 90.0, 120.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(old)
            added_log = work / "added-log.csv"
            write_csv(
                added_log,
                [row(theta, -0.5) for theta in (60.0, 90.0, 120.0)],
            )
            write_manifest(added_log)
            added_theta = work / "added-theta.csv"
            write_csv(
                added_theta,
                [
                    row(theta, log_m)
                    for theta in (45.0, 135.0)
                    for log_m in (-1.0, -0.5, 0.0)
                ],
            )
            write_manifest(added_theta)
            shards = [old, added_log, added_theta]
            table = work / "selected.csv"
            arguments = Namespace(
                shards=[str(path) for path in reversed(shards)],
                output=str(table),
                kind="table",
                theta_grid="45,60:120:30,135",
                log10_m_grid="-1:0:0.5",
            )
            self.assertEqual(FREEZE.merge_command(arguments), 0)

            with table.open(newline="", encoding="utf-8") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 15)
            self.assertEqual(
                sorted({float(item["theta_deg"]) for item in selected}),
                [45.0, 60.0, 90.0, 120.0, 135.0],
            )
            self.assertEqual(
                sorted(
                    {float(item["log10_viscosity_ratio"]) for item in selected}
                ),
                [-1.0, -0.5, 0.0],
            )
            manifest = json.loads(
                table.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["selection"]["input_rows"], 19)
            self.assertEqual(manifest["selection"]["selected_rows"], 15)
            self.assertEqual(manifest["selection"]["discarded_rows"], 4)
            self.assertEqual(len(manifest["inputs"]), 3)
            interpolation = manifest["runtime_interpolation"]
            self.assertEqual(
                interpolation["theta_deg"],
                {"stencil_nodes": 4, "polynomial_degree": 3},
            )
            self.assertEqual(
                interpolation["log10_M"],
                {"stencil_nodes": 5, "polynomial_degree": 4},
            )

            csv_bytes = table.read_bytes()
            manifest_text = table.with_suffix(".manifest.json").read_text(
                encoding="utf-8"
            )
            arguments.shards = [str(path) for path in shards]
            self.assertEqual(FREEZE.merge_command(arguments), 0)
            self.assertEqual(table.read_bytes(), csv_bytes)
            self.assertEqual(
                table.with_suffix(".manifest.json").read_text(encoding="utf-8"),
                manifest_text,
            )

    def test_merge_validates_excluded_input_before_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            selected = work / "selected.csv"
            write_csv(
                selected,
                [
                    row(theta, log_m)
                    for theta in (60.0, 120.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(selected)
            excluded = work / "excluded.csv"
            write_csv(
                excluded,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(excluded)
            with excluded.open("a", encoding="utf-8") as stream:
                stream.write("\n")

            with self.assertRaisesRegex(ValueError, "CSV hash"):
                FREEZE.merge_command(
                    Namespace(
                        shards=[str(selected), str(excluded)],
                        output=str(work / "table.csv"),
                        kind="table",
                        theta_grid="60,120",
                        log10_m_grid="-1,0",
                    )
                )

    def test_merge_rejects_missing_target_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shard = work / "shard.csv"
            write_csv(
                shard,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(shard)
            with self.assertRaisesRegex(ValueError, "target grid is missing 2"):
                FREEZE.merge_command(
                    Namespace(
                        shards=[str(shard)],
                        output=str(work / "table.csv"),
                        kind="table",
                        theta_grid="30,90,150",
                        log10_m_grid="-1,0",
                    )
                )

    def test_merge_rejects_conflicting_input_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            first = work / "first.csv"
            second = work / "second.csv"
            first_rows = [
                row(theta, log_m)
                for theta in (30.0, 150.0)
                for log_m in (-1.0, 0.0)
            ]
            second_rows = [dict(item) for item in first_rows]
            second_rows[0]["Q"] += 0.25
            write_csv(first, first_rows)
            write_manifest(first)
            write_csv(second, second_rows)
            write_manifest(second)
            with self.assertRaisesRegex(
                ValueError, "duplicate or conflicting input node"
            ):
                FREEZE.merge_command(
                    Namespace(
                        shards=[str(first), str(second)],
                        output=str(work / "table.csv"),
                        kind="table",
                        theta_grid="30,150",
                        log10_m_grid="-1,0",
                    )
                )

    def test_merge_rejects_range_that_misses_stop(self):
        with self.assertRaisesRegex(ValueError, "does not land exactly"):
            FREEZE._parse_target_grid("30:150:50", "theta", 12)

    def test_cubic_quartic_stencil_reproduces_tensor_polynomial(self):
        theta_grid = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0]
        log_grid = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]

        def polynomial(theta, log_m):
            scaled_theta = theta / 100.0
            theta_part = (
                0.3 * scaled_theta**3
                - 0.2 * scaled_theta**2
                + 0.7 * scaled_theta
                - 0.1
            )
            log_part = (
                -0.04 * log_m**4
                + 0.2 * log_m**3
                - 0.1 * log_m**2
                + 0.5 * log_m
                + 1.2
            )
            return theta_part * log_part

        values = {}
        for theta in theta_grid:
            for log_m in log_grid:
                item = row(theta, log_m)
                item["Q"] = polynomial(theta, log_m)
                values[(theta, log_m)] = item

        theta_query = 82.0
        log_query = -1.2
        interpolated, _, _, _, weights = FREEZE._interpolation_details(
            theta_grid, log_grid, values, theta_query, log_query
        )
        self.assertEqual(len(weights), 20)
        self.assertAlmostEqual(
            interpolated, polynomial(theta_query, log_query), places=13
        )

    def test_check_rejects_stale_stencil_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shard = work / "shard.csv"
            write_csv(
                shard,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(shard)
            table = work / "table.csv"
            FREEZE.merge_command(
                Namespace(shards=[str(shard)], output=str(table))
            )
            manifest_path = table.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_interpolation"]["log10_M"]["stencil_nodes"] = 4
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            checkpoints = work / "checkpoints.csv"
            write_csv(checkpoints, [row(90.0, -0.5)])
            write_manifest(checkpoints)
            with self.assertRaisesRegex(
                ValueError, "interpolation metadata differs"
            ):
                FREEZE.check_command(
                    Namespace(
                        table=str(table),
                        checkpoints=str(checkpoints),
                        output=str(work / "audit.csv"),
                        tolerance_observed_Q=1.0e-3,
                        tolerance_error_budget_Q=3.0e-3,
                        tolerance_right_angle_Q=1.0e-3,
                        tolerance_symmetry_Q=1.0e-3,
                    )
                )

    def test_check_rejects_missing_cell_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            table = work / "table.csv"
            rows = [
                row(theta, log_m)
                for theta in (30.0, 60.0, 90.0)
                for log_m in (-1.0, 0.0)
            ]
            write_csv(table, rows)
            write_manifest(table, 30.0)
            checkpoints = work / "checkpoints.csv"
            write_csv(checkpoints, [row(45.0, -0.5)])
            write_manifest(checkpoints, 45.0)
            with self.assertRaisesRegex(ValueError, "do not cover every table cell"):
                FREEZE.check_command(
                    Namespace(
                        table=str(table),
                        checkpoints=str(checkpoints),
                        output=str(work / "audit.csv"),
                        tolerance_observed_Q=1.0e-3,
                        tolerance_error_budget_Q=3.0e-3,
                        tolerance_right_angle_Q=1.0e-3,
                        tolerance_symmetry_Q=1.0e-3,
                    )
                )

    def test_merge_rejects_positive_stored_log_m(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shard = work / "positive-half-plane.csv"
            write_csv(
                shard,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0, 1.0)
                ],
            )
            write_manifest(shard)
            with self.assertRaisesRegex(ValueError, "at or below zero"):
                FREEZE.merge_command(
                    Namespace(shards=[str(shard)], output=str(work / "table.csv"))
                )

    def test_merge_rejects_asymmetric_theta_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shard = work / "asymmetric.csv"
            write_csv(
                shard,
                [
                    row(theta, log_m)
                    for theta in (30.0, 140.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(shard)
            with self.assertRaisesRegex(ValueError, "reflection-symmetric"):
                FREEZE.merge_command(
                    Namespace(shards=[str(shard)], output=str(work / "table.csv"))
                )

    def test_merge_rejects_unconverged_first_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shard = work / "unconverged.csv"
            write_csv(
                shard,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(shard, all_converged=False)
            with self.assertRaisesRegex(ValueError, "not convergence certified"):
                FREEZE.merge_command(
                    Namespace(shards=[str(shard)], output=str(work / "table.csv"))
                )

    def test_check_rejects_off_centre_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            table = work / "table.csv"
            write_csv(
                table,
                [
                    row(theta, log_m)
                    for theta in (30.0, 60.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(table)
            checkpoints = work / "checkpoints.csv"
            write_csv(checkpoints, [row(40.0, -0.5)])
            write_manifest(checkpoints)
            with self.assertRaisesRegex(ValueError, "table-cell centre"):
                FREEZE.check_command(
                    Namespace(
                        table=str(table),
                        checkpoints=str(checkpoints),
                        output=str(work / "audit.csv"),
                        tolerance_observed_Q=1.0e-3,
                        tolerance_error_budget_Q=3.0e-3,
                        tolerance_right_angle_Q=1.0e-3,
                        tolerance_symmetry_Q=1.0e-3,
                    )
                )

    def test_check_fails_wrong_right_angle_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            table = work / "table.csv"
            write_csv(
                table,
                [
                    row(theta, log_m)
                    for theta in (60.0, 90.0, 120.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(table)
            checkpoints = work / "checkpoints.csv"
            write_csv(
                checkpoints,
                [row(theta, -0.5) for theta in (75.0, 105.0)],
            )
            write_manifest(checkpoints)
            self.assertEqual(
                FREEZE.check_command(
                    Namespace(
                        table=str(table),
                        checkpoints=str(checkpoints),
                        output=str(work / "audit.csv"),
                        tolerance_observed_Q=1.0e-14,
                        tolerance_error_budget_Q=3.0e-5,
                        tolerance_right_angle_Q=1.0e-3,
                        tolerance_symmetry_Q=1.0e-3,
                    )
                ),
                1,
            )

    def test_emit_rejects_stale_table_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            table = work / "table.csv"
            write_csv(
                table,
                [
                    row(theta, log_m)
                    for theta in (30.0, 150.0)
                    for log_m in (-1.0, 0.0)
                ],
            )
            write_manifest(table)
            with table.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "CSV hash"):
                FREEZE.emit_c_command(
                    Namespace(table=str(table), output=str(work / "table-data.h"))
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
