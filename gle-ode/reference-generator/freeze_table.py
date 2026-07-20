#!/usr/bin/env python3
"""Merge converged FEM shards, audit interpolation, and emit C table data.

This helper has no third-party dependencies.  The expensive Stokes solves live
in ``generate.py``; this file handles only deterministic post-processing of
their CSV and manifest outputs.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_COLUMNS = {
    "theta_deg",
    "theta_rad",
    "log10_viscosity_ratio",
    "viscosity_ratio",
    "Q",
    "converged",
    "estimated_error_Q",
    "slope_relative_error",
    "plateau_half_drift_Q",
    "max_linear_system_backward_error",
    "max_constraint_relative_error",
    "generator_source_sha256",
}

THETA_STENCIL_NODES = 4
LOG10_M_STENCIL_NODES = 5


def _interpolation_metadata() -> Dict[str, object]:
    """Return metadata derived from the stencil used by the audit."""

    return {
        "method": "local tensor-product Lagrange",
        "interpolated_quantity": "Q",
        "theta_deg": {
            "stencil_nodes": THETA_STENCIL_NODES,
            "polynomial_degree": THETA_STENCIL_NODES - 1,
        },
        "log10_M": {
            "stencil_nodes": LOG10_M_STENCIL_NODES,
            "polynomial_degree": LOG10_M_STENCIL_NODES - 1,
        },
    }


def _interpolation_summary() -> str:
    return (
        "local tensor-product Lagrange interpolation using "
        f"{THETA_STENCIL_NODES} theta_deg nodes "
        f"(degree {THETA_STENCIL_NODES - 1}) and "
        f"{LOG10_M_STENCIL_NODES} log10(M) nodes "
        f"(degree {LOG10_M_STENCIL_NODES - 1})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def _write_csv_atomic(
    path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def _float(row: Dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in table row")
    return value


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty table")
    for row in rows:
        if not _is_true(row["converged"]):
            raise ValueError(f"{path}: contains an unconverged row")
        for key in REQUIRED_COLUMNS.difference(
            {"converged", "generator_source_sha256"}
        ):
            _float(row, key)
        source_sha = row["generator_source_sha256"]
        if len(source_sha) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha
        ):
            raise ValueError(f"{path}: invalid generator source SHA-256")
    return fields, rows


def _read_manifest(path: Path) -> Dict[str, object]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"missing shard manifest {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _same_configuration(
    reference: Dict[str, object], candidate: Dict[str, object], path: Path
) -> None:
    # runtime_interpolation is freezer output metadata, not a property of the
    # FEM solve. A frozen dataset can therefore be refined with raw shards.
    for key in (
        "generator_version",
        "generator_source_sha256",
        "interpolated_quantity",
        "method",
        "mesh",
        "fit_window",
        "tolerance_Q",
        "tolerance_slope",
        "tolerance_linear",
        "convergence_checks",
    ):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"{path}: manifest differs in {key}")
    if not candidate.get("all_converged", False):
        raise ValueError(f"{path}: manifest is not convergence certified")


def _validate_manifest(
    path: Path, manifest: Dict[str, object], rows: Sequence[Dict[str, str]]
) -> None:
    if not manifest.get("all_converged", False):
        raise ValueError(f"{path}: manifest is not convergence certified")
    if manifest.get("table") != path.name:
        raise ValueError(f"{path}: manifest table name does not match the CSV")
    if manifest.get("rows") != len(rows):
        raise ValueError(f"{path}: manifest row count does not match the CSV")
    theta, log_m = _grid(rows)
    manifest_theta = manifest.get("theta_deg_grid")
    if not isinstance(manifest_theta, list) or len(manifest_theta) != len(theta) or any(
        not math.isclose(float(expected), actual, abs_tol=1.0e-12)
        for expected, actual in zip(manifest_theta, theta)
    ):
        raise ValueError(f"{path}: manifest theta grid does not match the CSV")
    manifest_log_m = manifest.get("log10_M_grid")
    if not isinstance(manifest_log_m, list) or len(manifest_log_m) != len(log_m) or any(
        not math.isclose(float(expected), actual, abs_tol=1.0e-14)
        for expected, actual in zip(manifest_log_m, log_m)
    ):
        raise ValueError(f"{path}: manifest viscosity grid does not match the CSV")
    source_hashes = {row["generator_source_sha256"] for row in rows}
    if source_hashes != {manifest.get("generator_source_sha256")}:
        raise ValueError(f"{path}: generator source hash is inconsistent")
    generator_path = Path(__file__).with_name("generate.py")
    if manifest.get("generator_source_sha256") != _sha256(generator_path):
        raise ValueError(f"{path}: data came from a different generator source")
    expected_table_hash = manifest.get("table_sha256")
    if expected_table_hash != _sha256(path):
        raise ValueError(f"{path}: CSV hash does not match its manifest")


def _grid(rows: Iterable[Dict[str, str]]) -> Tuple[List[float], List[float]]:
    theta = sorted({_float(row, "theta_deg") for row in rows})
    log_m = sorted({_float(row, "log10_viscosity_ratio") for row in rows})
    return theta, log_m


def _assert_rectangular(
    rows: Sequence[Dict[str, str]], theta: Sequence[float], log_m: Sequence[float]
) -> Dict[Tuple[float, float], Dict[str, str]]:
    indexed: Dict[Tuple[float, float], Dict[str, str]] = {}
    for row in rows:
        key = (_float(row, "theta_deg"), _float(row, "log10_viscosity_ratio"))
        if key in indexed:
            raise ValueError(f"duplicate table node theta={key[0]}, log10(M)={key[1]}")
        indexed[key] = row
    expected = {(angle, ratio) for angle in theta for ratio in log_m}
    missing = expected.difference(indexed)
    extra = set(indexed).difference(expected)
    if missing or extra:
        raise ValueError(
            f"table is not rectangular: {len(missing)} missing, {len(extra)} extra"
        )
    return indexed


def _max_column(rows: Sequence[Dict[str, str]], key: str) -> float:
    return max(_float(row, key) for row in rows)


def _normalise_coordinate(value: float, digits: int) -> float:
    normalised = round(value, digits)
    return 0.0 if normalised == 0.0 else normalised


def _parse_target_grid(
    expression: str, label: str, digits: int
) -> List[float]:
    """Parse an exact comma/range grid and return canonical coordinates."""

    if not expression.strip():
        raise ValueError(f"empty {label} target grid")
    decimal_values: List[Decimal] = []
    for item in expression.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty item in {label} target grid")
        parts = item.split(":")
        try:
            values = [Decimal(part.strip()) for part in parts]
        except InvalidOperation as error:
            raise ValueError(
                f"invalid number in {label} target grid: {item}"
            ) from error
        if any(not value.is_finite() for value in values):
            raise ValueError(f"non-finite number in {label} target grid: {item}")
        if len(values) == 1:
            decimal_values.append(values[0])
            continue
        if len(values) != 3:
            raise ValueError(
                f"{label} range must be start:stop:step: {item}"
            )
        start, stop, step = values
        if step == 0 or (stop - start) * step < 0:
            raise ValueError(f"invalid {label} range: {item}")
        steps = (stop - start) / step
        integral_steps = steps.to_integral_value()
        if steps != integral_steps:
            raise ValueError(
                f"{label} range does not land exactly on its stop: {item}"
            )
        count = int(integral_steps) + 1
        if count > 100000:
            raise ValueError(f"{label} range is unreasonably large: {item}")
        decimal_values.extend(start + index * step for index in range(count))

    parsed = [
        _normalise_coordinate(float(value), digits) for value in decimal_values
    ]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate coordinate in {label} target grid")
    return sorted(parsed)


def merge_command(args: argparse.Namespace) -> int:
    dataset_kind = getattr(args, "kind", "table")
    paths = sorted(
        (Path(item) for item in args.shards),
        key=lambda path: (path.name, str(path)),
    )
    fields: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []
    manifests: List[Dict[str, object]] = []
    for path in paths:
        shard_fields, shard_rows = _read_csv(path)
        if fields is None:
            fields = shard_fields
        elif fields != shard_fields:
            raise ValueError(f"{path}: CSV columns differ from the first shard")
        rows.extend(shard_rows)
        manifest = _read_manifest(path)
        _validate_manifest(path, manifest, shard_rows)
        shard_theta, shard_log_m = _grid(shard_rows)
        _assert_rectangular(shard_rows, shard_theta, shard_log_m)
        manifests.append(manifest)
    if fields is None:
        raise ValueError("no shards supplied")
    for path, manifest in zip(paths[1:], manifests[1:]):
        _same_configuration(manifests[0], manifest, path)

    # The generator reports theta in radians and converts it back for the CSV;
    # remove harmless binary round trips such as 119.99999999999999 before
    # freezing interpolation coordinates.
    for row in rows:
        row["theta_deg"] = (
            f"{_normalise_coordinate(_float(row, 'theta_deg'), 12):.17g}"
        )
        normalised_log_m = _normalise_coordinate(
            _float(row, "log10_viscosity_ratio"), 14
        )
        row["log10_viscosity_ratio"] = (
            f"{normalised_log_m:.17g}"
        )

    indexed: Dict[Tuple[float, float], Dict[str, str]] = {}
    for row in rows:
        key = (_float(row, "theta_deg"), _float(row, "log10_viscosity_ratio"))
        if key in indexed:
            raise ValueError(
                "duplicate or conflicting input node "
                f"theta={key[0]}, log10(M)={key[1]}"
            )
        indexed[key] = row

    available_theta, available_log_m = _grid(rows)
    theta_expression = getattr(args, "theta_grid", None)
    log_expression = getattr(
        args, "log10_m_grid", getattr(args, "log10_M_grid", None)
    )
    theta = (
        _parse_target_grid(theta_expression, "theta", 12)
        if theta_expression is not None
        else available_theta
    )
    log_m = (
        _parse_target_grid(log_expression, "log10(M)", 14)
        if log_expression is not None
        else available_log_m
    )
    target_keys = {(angle, ratio) for angle in theta for ratio in log_m}
    missing = sorted(target_keys.difference(indexed))
    if missing:
        examples = ", ".join(
            f"({angle:g}, {ratio:g})" for angle, ratio in missing[:5]
        )
        suffix = "" if len(missing) <= 5 else ", ..."
        raise ValueError(
            f"target grid is missing {len(missing)} input point(s): "
            f"{examples}{suffix}"
        )
    input_row_count = len(rows)
    rows = [indexed[(angle, ratio)] for angle in theta for ratio in log_m]
    _assert_rectangular(rows, theta, log_m)
    if len(theta) < 2 or len(log_m) < 2:
        raise ValueError("the frozen table needs at least two nodes per dimension")
    if any(value > 0.0 for value in log_m):
        raise ValueError("the stored log10(M) grid must lie at or below zero")
    if dataset_kind == "table" and log_m[-1] != 0.0:
        raise ValueError("the table-node log10(M) grid must end at zero")
    if dataset_kind == "checkpoints" and log_m[-1] >= 0.0:
        raise ValueError("checkpoint log10(M) values must lie inside table cells")
    if not math.isclose(theta[0] + theta[-1], 180.0, abs_tol=1.0e-12):
        raise ValueError("the theta grid bounds must be reflection-symmetric about 90 deg")
    theta_set = set(theta)
    if any(round(180.0 - angle, 12) not in theta_set for angle in theta):
        raise ValueError("every theta node must have a phase-reflected partner")
    rows.sort(
        key=lambda row: (
            _float(row, "theta_deg"),
            _float(row, "log10_viscosity_ratio"),
        )
    )
    output = Path(args.output)
    _write_csv_atomic(output, fields, rows)

    source = manifests[0]
    manifest = {
        "generator_version": source["generator_version"],
        "generator_source_sha256": source["generator_source_sha256"],
        "schema": source.get("schema", "table.schema.json"),
        "table": output.name,
        "table_sha256": _sha256(output),
        "interpolated_quantity": "Q",
        "runtime_interpolation": _interpolation_metadata(),
        "c_reconstruction": "log(c) = 1 + log(sin(theta_rad)) - Q",
        "method": source["method"],
        "dataset_kind": (
            "table_nodes" if dataset_kind == "table" else
            "interpolation_checkpoints"
        ),
        "rows": len(rows),
        "all_converged": True,
        "theta_deg_grid": theta,
        "log10_M_grid": log_m,
        "selection": {
            "applied": theta_expression is not None or log_expression is not None,
            "input_rows": input_row_count,
            "selected_rows": len(rows),
            "discarded_rows": input_row_count - len(rows),
            "theta_deg_grid": theta,
            "log10_M_grid": log_m,
        },
        "mesh": source["mesh"],
        "fit_window": source["fit_window"],
        "tolerance_Q": source["tolerance_Q"],
        "tolerance_slope": source["tolerance_slope"],
        "tolerance_linear": source["tolerance_linear"],
        "convergence_checks": source["convergence_checks"],
        "max_errors": {
            "estimated_error_Q": _max_column(rows, "estimated_error_Q"),
            "slope_relative_error": _max_column(rows, "slope_relative_error"),
            "plateau_half_drift_Q": _max_column(rows, "plateau_half_drift_Q"),
            "linear_system_backward_error": _max_column(
                rows, "max_linear_system_backward_error"
            ),
            "constraint_relative_error": _max_column(
                rows, "max_constraint_relative_error"
            ),
        },
        "shards": sorted(path.name for path in paths),
        "inputs": sorted(
            (
                {
                    "table": path.name,
                    "table_sha256": manifest["table_sha256"],
                    "rows": manifest["rows"],
                }
                for path, manifest in zip(paths, manifests)
            ),
            key=lambda item: (str(item["table"]), str(item["table_sha256"])),
        ),
        "references": source.get("references", []),
        "failures": [],
    }
    if dataset_kind == "table":
        manifest["phase_exchange_extension"] = {
            "identity": "Q(theta,M) = Q(pi-theta,1/M)",
            "stored_log10_M_domain": [log_m[0], log_m[-1]],
            "runtime_log10_M_domain": [log_m[0], -log_m[0]],
        }
    _write_text_atomic(
        output.with_suffix(".manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"wrote {output} ({len(theta)} x {len(log_m)} = {len(rows)} nodes)"
    )
    return 0


def _bracket(grid: Sequence[float], query: float) -> Tuple[int, float]:
    if query < grid[0] or query > grid[-1]:
        raise ValueError(f"query {query} lies outside [{grid[0]}, {grid[-1]}]")
    if query == grid[-1]:
        return len(grid) - 2, 1.0
    index = bisect.bisect_right(grid, query) - 1
    index = max(0, min(index, len(grid) - 2))
    weight = (query - grid[index]) / (grid[index + 1] - grid[index])
    return index, weight


def _lagrange_stencil(
    grid: Sequence[float], cell: int, query: float, requested_nodes: int
) -> Tuple[List[int], List[float]]:
    """Return a local Lagrange stencil and its basis at ``query``."""

    if requested_nodes < 2:
        raise ValueError("a Lagrange stencil needs at least two nodes")
    count = min(requested_nodes, len(grid))
    start = cell - (count // 2 - 1)
    start = max(0, min(start, len(grid) - count))
    indices = list(range(start, start + count))
    weights = []
    for i in indices:
        numerator = 1.0
        denominator = 1.0
        for j in indices:
            if i == j:
                continue
            numerator *= query - grid[j]
            denominator *= grid[i] - grid[j]
        weights.append(numerator / denominator)
    return indices, weights


def _interpolation_details(
    theta_grid: Sequence[float],
    log_grid: Sequence[float],
    values: Dict[Tuple[float, float], Dict[str, str]],
    theta_deg: float,
    log10_m: float,
) -> Tuple[float, float, Tuple[int, int], Tuple[float, float], Tuple[float, ...]]:
    if log10_m > 0.0:
        theta_deg = 180.0 - theta_deg
        log10_m = -log10_m
    i, wt = _bracket(theta_grid, theta_deg)
    j, wm = _bracket(log_grid, log10_m)
    theta_indices, theta_weights = _lagrange_stencil(
        theta_grid, i, theta_deg, THETA_STENCIL_NODES
    )
    log_indices, log_weights = _lagrange_stencil(
        log_grid, j, log10_m, LOG10_M_STENCIL_NODES
    )
    weighted_nodes = [
        (
            theta_weights[local_i] * log_weights[local_j],
            values[(theta_grid[node_i], log_grid[node_j])],
        )
        for local_i, node_i in enumerate(theta_indices)
        for local_j, node_j in enumerate(log_indices)
    ]
    q = sum(weight * _float(node, "Q") for weight, node in weighted_nodes)
    # Lagrange weights may be negative. Absolute weights give the conservative
    # linear propagation of the recorded node sensitivity estimates.
    node_uncertainty = sum(
        abs(weight) * _float(node, "estimated_error_Q")
        for weight, node in weighted_nodes
    )
    weights = tuple(weight for weight, _ in weighted_nodes)
    return q, node_uncertainty, (i, j), (wt, wm), weights


def _interpolate_Q(
    theta_grid: Sequence[float],
    log_grid: Sequence[float],
    values: Dict[Tuple[float, float], Dict[str, str]],
    theta_deg: float,
    log10_m: float,
) -> float:
    return _interpolation_details(
        theta_grid, log_grid, values, theta_deg, log10_m
    )[0]


def _corrected_right_angle_Q(viscosity_ratio: float) -> float:
    """Published equal-slip right-angle authority (Luo--Gao, Eq. 4.14)."""

    gamma_e = 0.57721566490153286061
    h_a = 4.0 * (gamma_e - math.log(2.0)) / math.pi
    h_b = -1.539
    m = viscosity_ratio
    numerator = math.pi * (math.pi**2 - 4.0) * (
        (m - 1.0) ** 2 * h_a + 4.0 * m * h_b
    )
    denominator = 4.0 * math.pi**2 * (m + 1.0) ** 2 - 16.0 * (m - 1.0) ** 2
    return 1.0 + numerator / denominator


def check_command(args: argparse.Namespace) -> int:
    table_path = Path(args.table)
    checkpoint_path = Path(args.checkpoints)
    if table_path.resolve() == checkpoint_path.resolve():
        raise ValueError("the interpolation checkpoints must be independent of the table")
    _, table_rows = _read_csv(table_path)
    theta, log_m = _grid(table_rows)
    indexed = _assert_rectangular(table_rows, theta, log_m)
    table_manifest = _read_manifest(table_path)
    _validate_manifest(table_path, table_manifest, table_rows)
    recorded_interpolation = table_manifest.get("runtime_interpolation")
    if (
        recorded_interpolation is not None
        and recorded_interpolation != _interpolation_metadata()
    ):
        raise ValueError(
            "table manifest interpolation metadata differs from this freezer"
        )
    _, checks = _read_csv(checkpoint_path)
    checkpoint_manifest = _read_manifest(checkpoint_path)
    _validate_manifest(checkpoint_path, checkpoint_manifest, checks)
    _same_configuration(table_manifest, checkpoint_manifest, checkpoint_path)
    records = []
    observed_errors = []
    budget_errors = []
    covered_cells = set()
    for row in checks:
        theta_query = _float(row, "theta_deg")
        log_query = _float(row, "log10_viscosity_ratio")
        q_reference = _float(row, "Q")
        (
            q_interpolated,
            node_uncertainty,
            cell,
            local_coordinates,
            _,
        ) = _interpolation_details(
            theta, log_m, indexed, theta_query, log_query
        )
        wt, wm = local_coordinates
        if not (
            math.isclose(wt, 0.5, abs_tol=1.0e-12)
            and math.isclose(wm, 0.5, abs_tol=1.0e-12)
        ):
            raise ValueError(
                "every interpolation checkpoint must lie at a table-cell centre"
            )
        covered_cells.add(cell)
        observed_error = abs(q_interpolated - q_reference)
        reference_uncertainty = _float(row, "estimated_error_Q")
        budget_error = observed_error + reference_uncertainty + node_uncertainty
        observed_errors.append(observed_error)
        budget_errors.append(budget_error)
        records.append(
            {
                "theta_deg": f"{theta_query:.17g}",
                "log10_viscosity_ratio": f"{log_query:.17g}",
                "Q_reference": f"{q_reference:.17g}",
                "Q_interpolated": f"{q_interpolated:.17g}",
                "absolute_error_Q": f"{observed_error:.17g}",
                "reference_estimated_error_Q": row["estimated_error_Q"],
                "absolute_weighted_node_sensitivity_Q": (
                    f"{node_uncertainty:.17g}"
                ),
                "checkpoint_error_budget_Q": f"{budget_error:.17g}",
                "table_cell_theta_index": cell[0],
                "table_cell_log10_M_index": cell[1],
                "reference_converged": row["converged"],
            }
        )
    required_cells = {
        (i, j) for i in range(len(theta) - 1) for j in range(len(log_m) - 1)
    }
    missing_cells = sorted(required_cells.difference(covered_cells))
    if missing_cells:
        raise ValueError(
            "interpolation checkpoints do not cover every table cell; "
            f"{len(missing_cells)} cells are missing"
        )
    worst_observed = max(
        range(len(observed_errors)), key=observed_errors.__getitem__
    )
    worst_budget = max(
        range(len(budget_errors)), key=budget_errors.__getitem__
    )
    right_angle_rows = [
        row for row in table_rows
        if math.isclose(_float(row, "theta_deg"), 90.0, abs_tol=1.0e-12)
    ]
    right_angle_errors = [
        abs(
            _float(row, "Q")
            - _corrected_right_angle_Q(_float(row, "viscosity_ratio"))
        )
        for row in right_angle_rows
    ]
    worst_right_angle = (
        max(range(len(right_angle_errors)), key=right_angle_errors.__getitem__)
        if right_angle_errors else None
    )
    max_right_angle_error = (
        right_angle_errors[worst_right_angle]
        if worst_right_angle is not None else 0.0
    )
    equal_viscosity_errors = []
    if 0.0 in log_m:
        theta_set = set(theta)
        for angle in theta:
            reflected = round(180.0 - angle, 12)
            if reflected not in theta_set:
                raise ValueError(
                    "the M=1 column needs a phase-reflected theta partner"
                )
            if angle <= reflected:
                equal_viscosity_errors.append(
                    abs(
                        _float(indexed[(angle, 0.0)], "Q")
                        - _float(indexed[(reflected, 0.0)], "Q")
                    )
                )
    max_equal_viscosity_error = max(equal_viscosity_errors, default=0.0)
    stats = {
        "checkpoints": len(observed_errors),
        "required_cells": len(required_cells),
        "covered_cells": len(covered_cells),
        "max_absolute_error_Q": observed_errors[worst_observed],
        "mean_absolute_error_Q": statistics.fmean(observed_errors),
        "rms_error_Q": math.sqrt(
            statistics.fmean(error * error for error in observed_errors)
        ),
        "max_checkpoint_error_budget_Q": budget_errors[worst_budget],
        "tolerance_observed_Q": args.tolerance_observed_Q,
        "tolerance_error_budget_Q": args.tolerance_error_budget_Q,
        "right_angle_anchor_nodes": len(right_angle_rows),
        "max_right_angle_anchor_error_Q": max_right_angle_error,
        "tolerance_right_angle_anchor_Q": args.tolerance_right_angle_Q,
        "max_equal_viscosity_symmetry_error_Q": max_equal_viscosity_error,
        "tolerance_equal_viscosity_symmetry_Q": args.tolerance_symmetry_Q,
        "interpolation": _interpolation_metadata(),
        "scope": (
            f"cell-centre empirical {_interpolation_summary()} "
            "discrepancy plus FEM sensitivity estimates; not a global "
            "mathematical error bound"
        ),
        "passed": (
            observed_errors[worst_observed] <= args.tolerance_observed_Q
            and budget_errors[worst_budget] <= args.tolerance_error_budget_Q
            and max_right_angle_error <= args.tolerance_right_angle_Q
            and max_equal_viscosity_error <= args.tolerance_symmetry_Q
        ),
        "worst_observed_case": {
            "theta_deg": _float(checks[worst_observed], "theta_deg"),
            "log10_viscosity_ratio": _float(
                checks[worst_observed], "log10_viscosity_ratio"
            ),
            "reference_Q": _float(checks[worst_observed], "Q"),
            "interpolated_Q": _interpolate_Q(
                theta,
                log_m,
                indexed,
                _float(checks[worst_observed], "theta_deg"),
                _float(checks[worst_observed], "log10_viscosity_ratio"),
            ),
        },
        "worst_error_budget_case": {
            "theta_deg": _float(checks[worst_budget], "theta_deg"),
            "log10_viscosity_ratio": _float(
                checks[worst_budget], "log10_viscosity_ratio"
            ),
            "checkpoint_error_budget_Q": budget_errors[worst_budget],
        },
    }
    if worst_right_angle is not None:
        right_row = right_angle_rows[worst_right_angle]
        stats["worst_right_angle_anchor_case"] = {
            "viscosity_ratio": _float(right_row, "viscosity_ratio"),
            "reference_Q": _corrected_right_angle_Q(
                _float(right_row, "viscosity_ratio")
            ),
            "fem_Q": _float(right_row, "Q"),
        }
    output = Path(args.output)
    _write_csv_atomic(output, list(records[0]), records)
    stats["table_sha256"] = _sha256(table_path)
    stats["checkpoint_file"] = checkpoint_path.name
    stats["checkpoint_sha256"] = _sha256(checkpoint_path)
    stats["audit_file"] = output.name
    stats["audit_sha256"] = _sha256(output)
    stats["generator_source_sha256"] = table_manifest[
        "generator_source_sha256"
    ]
    manifest_path = table_path.with_suffix(".manifest.json")
    manifest = table_manifest
    manifest["interpolation_validation"] = stats
    _write_text_atomic(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["passed"] else 1


def _c_array(values: Sequence[float], indent: str = "  ") -> str:
    return "{\n" + ",\n".join(
        indent + f"{value:.17g}" for value in values
    ) + "\n}"


def emit_c_command(args: argparse.Namespace) -> int:
    table_path = Path(args.table)
    _, rows = _read_csv(table_path)
    theta, log_m = _grid(rows)
    indexed = _assert_rectangular(rows, theta, log_m)
    manifest = json.loads(
        table_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    _validate_manifest(table_path, manifest, rows)
    validation = manifest.get("interpolation_validation")
    if not isinstance(validation, dict) or not validation.get("passed", False):
        raise ValueError("table manifest has no passing interpolation validation")
    if validation.get("interpolation") != _interpolation_metadata():
        raise ValueError("interpolation validation uses a different stencil")
    if validation.get("table_sha256") != _sha256(table_path):
        raise ValueError("interpolation validation refers to a stale table")
    for file_key, hash_key in (
        ("checkpoint_file", "checkpoint_sha256"),
        ("audit_file", "audit_sha256"),
    ):
        evidence = table_path.parent / str(validation.get(file_key, ""))
        if not evidence.is_file() or validation.get(hash_key) != _sha256(evidence):
            raise ValueError(f"interpolation validation evidence is stale: {evidence}")
    q_rows = [
        [_float(indexed[(angle, ratio)], "Q") for ratio in log_m]
        for angle in theta
    ]
    # The exact M=1 phase-exchange identity is stronger than the tiny FEM
    # asymmetry left by independent sector meshes. Average reflected pairs in
    # the emitted runtime data; the raw FEM values remain untouched in CSV.
    max_m1_correction = 0.0
    if log_m[-1] == 0.0:
        j = len(log_m) - 1
        for i in range((len(theta) + 1) // 2):
            k = len(theta) - 1 - i
            average = 0.5 * (q_rows[i][j] + q_rows[k][j])
            max_m1_correction = max(
                max_m1_correction,
                abs(q_rows[i][j] - average),
                abs(q_rows[k][j] - average),
            )
            q_rows[i][j] = q_rows[k][j] = average
    q_text = "{\n" + ",\n".join(
        "  " + _c_array(row, indent="    ").replace("\n", "\n  ").rstrip()
        for row in q_rows
    ) + "\n}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = f'''/**
# gle-slip-table-data.h -- generated constant-slip Stokes reference data

Generated from `{table_path.name}` by `gle-ode/reference-generator/freeze_table.py`.
Generator source SHA-256: `{manifest['generator_source_sha256']}`.
Frozen table SHA-256: `{manifest['table_sha256']}`.
The stored half-plane uses ${log_m[0]:.8g}\\leq\\log_{{10}}M\\leq0$; the runtime applies
$Q(\\theta,M)=Q(\\pi-\\theta,1/M)$ for $M>1$. Interpolate $Q$, never $c$.
Validated runtime interpolation uses {_interpolation_summary()}.
Reflected $M=1$ FEM pairs are averaged to impose exact phase symmetry; the
maximum change is {max_m1_correction:.8g} in $Q$.
*/

#ifndef GLE_SLIP_TABLE_DATA_H
#define GLE_SLIP_TABLE_DATA_H

#define GLE_SLIP_TABLE_THETA_COUNT {len(theta)}
#define GLE_SLIP_TABLE_LOGM_COUNT {len(log_m)}
#define GLE_SLIP_TABLE_THETA_STENCIL_NODES {THETA_STENCIL_NODES}
#define GLE_SLIP_TABLE_LOGM_STENCIL_NODES {LOG10_M_STENCIL_NODES}

static const double gle_slip_table_theta_deg[GLE_SLIP_TABLE_THETA_COUNT] =
{_c_array(theta)};

static const double gle_slip_table_log10_m[GLE_SLIP_TABLE_LOGM_COUNT] =
{_c_array(log_m)};

static const double
gle_slip_table_Q[GLE_SLIP_TABLE_THETA_COUNT][GLE_SLIP_TABLE_LOGM_COUNT] =
{q_text};

static const double gle_slip_table_max_node_error_Q =
  {manifest['max_errors']['estimated_error_Q']:.17g};
static const double gle_slip_table_max_interpolation_error_Q =
  {validation['max_absolute_error_Q']:.17g};
static const double gle_slip_table_max_checkpoint_error_budget_Q =
  {validation['max_checkpoint_error_budget_Q']:.17g};
static const double gle_slip_table_max_m1_symmetrisation_Q =
  {max_m1_correction:.17g};

#endif /* GLE_SLIP_TABLE_DATA_H */
'''
    _write_text_atomic(output, text)
    print(f"wrote {output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="merge convergence-checked shards")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument(
        "--kind", choices=("table", "checkpoints"), default="table"
    )
    merge.add_argument(
        "--theta-grid",
        help="optional exact output theta grid (comma values/start:stop:step)",
    )
    merge.add_argument(
        "--log10-M-grid",
        dest="log10_m_grid",
        help="optional exact output log10(M) grid (comma values/start:stop:step)",
    )
    merge.set_defaults(function=merge_command)

    check = subparsers.add_parser(
        "check",
        help=f"check {_interpolation_summary()} against FEM checkpoints",
    )
    check.add_argument("--table", required=True)
    check.add_argument("--checkpoints", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--tolerance-observed-Q", type=float, default=1.0e-3)
    check.add_argument("--tolerance-error-budget-Q", type=float, default=3.0e-3)
    check.add_argument("--tolerance-right-angle-Q", type=float, default=1.0e-3)
    check.add_argument("--tolerance-symmetry-Q", type=float, default=1.0e-3)
    check.set_defaults(function=check_command)

    emit = subparsers.add_parser("emit-c", help="emit dependency-free C data")
    emit.add_argument("--table", required=True)
    emit.add_argument("--output", required=True)
    emit.set_defaults(function=emit_c_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
