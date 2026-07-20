#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#   "numpy==2.0.2",
#   "scipy==1.13.1",
# ]
# ///
"""Generate the one-phase Scott--Hocking matching reference.

This program solves Scott's public integral equation (2.1) for the wall-stress
function and evaluates his equation (2.11) for ``Qi``.  It is deliberately
separate from the two-phase FEM generator: changing this file cannot stale or
rewrite the finite-viscosity evidence.

The discretisation is a regularised Nyström method.  Subtracting the value at
the collocation point removes the logarithmic kernel singularity.  Analytic
kernel tails from Scott's Appendix B close the truncated interval.  The frozen
table is checked by a coarser solve, a wider-domain solve, Scott's published
Table 1, the analytic right-angle value, and direct solves at all interior
table-cell midpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import solve, toeplitz


METHOD = "scott-2020-regularised-nystrom-v1"
EULER_GAMMA = 0.57721566490153286061
RIGHT_ANGLE_QI = EULER_GAMMA - math.log(2.0)
LARGE_ANGLE_CONSTANT = RIGHT_ANGLE_QI - 2.0
ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_TABLE = ROOT / "data" / "scott-hocking-m0-nodes.csv"
DEFAULT_CHECKPOINTS = ROOT / "data" / "scott-hocking-m0-checkpoints.csv"
DEFAULT_MANIFEST = ROOT / "data" / "scott-hocking-m0.manifest.json"
DEFAULT_HEADER = REPOSITORY / "src-local" / "gle-slip-scott-data.h"
PUBLISHED_TABLE = ROOT / "data" / "scott-hocking-m0.csv"


@dataclass(frozen=True)
class SolveResult:
    theta_rad: float
    qi: float
    spacing: float
    radius: float
    series_terms: int
    relative_residual: float


@dataclass(frozen=True)
class ConvergedResult:
    primary: SolveResult
    coarse_qi: float
    wide_qi: float
    discretisation_delta_qi: float
    domain_delta_qi: float
    estimated_error_qi: float


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kernel_and_tail(
    theta: float, distances: np.ndarray, series_terms: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return Scott's ``L(rho)`` and ``K0(rho)`` for non-negative rho.

    The exponentially stable forms are equations (B.4) and (B.5).  Chunking
    bounds temporary memory when a large outer interval is needed near pi.
    ``L(0)`` is never requested: the regularised integrand vanishes there.
    """

    values = np.asarray(distances, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("kernel distances must be non-negative")
    harmonics = np.arange(1, series_terms + 1, dtype=float) * math.pi
    kernel = np.empty_like(values)
    tail = np.empty_like(values)
    block = 512
    for start in range(0, values.size, block):
        stop = min(start + block, values.size)
        rho = values[start:stop, None]
        minus = harmonics - theta
        plus = harmonics + theta
        decay_minus = np.exp(-(harmonics / theta - 1.0) * rho)
        decay_plus = np.exp(-(harmonics / theta + 1.0) * rho)
        kernel[start:stop] = np.sum(
            harmonics * (decay_minus / minus - decay_plus / plus), axis=1
        ) / (4.0 * theta)
        tail[start:stop] = np.sum(
            harmonics
            * (decay_minus / (minus * minus) - decay_plus / (plus * plus)),
            axis=1,
        ) / 4.0
    return kernel, tail


def _simpson(values: np.ndarray, spacing: float) -> float:
    intervals = values.size - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("composite Simpson rule needs a positive even interval count")
    weights = np.ones(values.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float((spacing / 3.0) * np.dot(weights, values))


def solve_qi(
    theta: float,
    *,
    spacing: float = 0.05,
    radius: float = 40.0,
    series_terms: int = 3000,
) -> SolveResult:
    """Solve Scott's integral equation and return ``Qi(theta)``.

    Writing ``int L k' = k/k_inf + int L (k' - k)`` makes the integrand
    continuous at the logarithmic singularity.  Beyond ``[-radius, radius]``
    the exact limiting values ``k=0`` and ``k=k_inf`` are integrated with
    ``K0`` rather than silently discarded.
    """

    if not (math.isfinite(theta) and 0.0 < theta < math.pi):
        raise ValueError("theta must lie strictly between zero and pi")
    if not (math.isfinite(spacing) and spacing > 0.0):
        raise ValueError("spacing must be positive")
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius must be positive")
    half_intervals = int(round(radius / spacing))
    if half_intervals < 4 or half_intervals % 2:
        raise ValueError("radius/spacing must be an even integer >= 4")
    if not math.isclose(
        half_intervals * spacing, radius, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("radius must be an integer multiple of spacing")
    if series_terms < 50:
        raise ValueError("series_terms must be at least 50")

    rho = np.arange(-half_intervals, half_intervals + 1, dtype=float) * spacing
    count = rho.size
    distances = np.arange(1, count, dtype=float) * spacing
    kernel_positive, _ = _kernel_and_tail(theta, distances, series_terms)
    first_column = np.empty(count)
    first_column[0] = 0.0
    first_column[1:] = kernel_positive
    matrix = spacing * toeplitz(first_column)
    quadrature_weights = np.ones(count)
    quadrature_weights[[0, -1]] = 0.5
    matrix *= quadrature_weights[None, :]

    sine = math.sin(theta)
    k_infinity = 2.0 * sine * sine / (theta - sine * math.cos(theta))
    base = np.exp(-rho) + 1.0 / k_infinity
    diagonal = np.diag_indices(count)
    matrix[diagonal] = base - np.sum(matrix, axis=1)

    _, left_tail = _kernel_and_tail(theta, rho + radius, series_terms)
    _, right_tail = _kernel_and_tail(theta, radius - rho, series_terms)
    matrix[diagonal] -= left_tail + right_tail
    rhs = 1.0 - k_infinity * right_tail

    # Row scaling removes the harmless exp(-rho) dynamic range from the linear
    # system.  It changes neither the collocation equations nor their solution.
    matrix /= base[:, None]
    rhs /= base
    wall_stress = solve(matrix, rhs, assume_a="gen", check_finite=False)
    # ``einsum(..., optimize=False)`` avoids a spurious Accelerate/BLAS
    # floating-point warning seen for the strongly row-scaled matrix while
    # retaining the ordinary double-precision residual calculation.
    residual = np.einsum(
        "ij,j->i", matrix, wall_stress, optimize=False
    ) - rhs
    residual_scale = (
        np.linalg.norm(matrix, ord=np.inf)
        * np.linalg.norm(wall_stress, ord=np.inf)
        + np.linalg.norm(rhs, ord=np.inf)
    )
    relative_residual = float(
        np.linalg.norm(residual, ord=np.inf) / max(residual_scale, np.finfo(float).tiny)
    )

    centre = half_intervals
    negative_integral = _simpson(wall_stress[: centre + 1], spacing)
    positive_gamma = wall_stress[centre:] - k_infinity
    positive_integral = _simpson(positive_gamma, spacing)
    qi = (negative_integral + positive_integral) / k_infinity
    if not (math.isfinite(qi) and math.isfinite(relative_residual)):
        raise RuntimeError("non-finite Scott--Hocking solve")
    return SolveResult(
        theta_rad=theta,
        qi=float(qi),
        spacing=spacing,
        radius=radius,
        series_terms=series_terms,
        relative_residual=relative_residual,
    )


def _primary_settings(theta: float) -> tuple[float, float]:
    if theta <= 0.075:
        return 0.025, 40.0
    if theta <= 2.8:
        return 0.05, 40.0
    if theta <= 3.0:
        return 0.05, 60.0
    return 0.10, 240.0


def solve_converged(theta: float, *, series_terms: int = 3000) -> ConvergedResult:
    """Return the production solve and two independent sensitivity solves."""

    spacing, radius = _primary_settings(theta)
    primary = solve_qi(
        theta, spacing=spacing, radius=radius, series_terms=series_terms
    )
    coarse = solve_qi(
        theta,
        spacing=2.0 * spacing,
        radius=radius,
        series_terms=series_terms,
    )
    if theta <= 2.8:
        wide_spacing, wide_radius = spacing, radius + 20.0
    elif theta <= 3.0:
        wide_spacing, wide_radius = 2.0 * spacing, 100.0
    else:
        # Compare R=240 and R=360 at the same coarse spacing. This isolates
        # the slowly decaying large-angle domain sensitivity without forming
        # a matrix larger than the primary solve.
        wide_spacing, wide_radius = 2.0 * spacing, 360.0
    wide = solve_qi(
        theta,
        spacing=wide_spacing,
        radius=wide_radius,
        series_terms=series_terms,
    )
    discretisation_delta = abs(primary.qi - coarse.qi)
    domain_delta = (
        abs(primary.qi - wide.qi)
        if theta <= 2.8
        else abs(coarse.qi - wide.qi)
    )
    estimated = max(discretisation_delta, domain_delta)
    return ConvergedResult(
        primary=primary,
        coarse_qi=coarse.qi,
        wide_qi=wide.qi,
        discretisation_delta_qi=discretisation_delta,
        domain_delta_qi=domain_delta,
        estimated_error_qi=estimated,
    )


def default_node_angles() -> list[float]:
    """Dense nonuniform grid, refined towards both singular angle limits."""

    values = [0.05, 0.075]
    values.extend(round(0.1 + 0.05 * index, 12) for index in range(59))
    # The regularised large-angle function has a shallow turning point near
    # 2.22 rad.  These three bisection nodes are the only refinements selected
    # by the independent midpoint tolerance on the otherwise 0.05-rad grid.
    values.extend(
        [2.2125, 2.225, 2.2375, 3.025, 3.05, 3.075, 3.1, 3.12, 0.5 * math.pi]
    )
    return sorted(set(values))


def _endpoint_augmented(
    angles: Sequence[float], qi: Sequence[float]
) -> tuple[list[float], list[float], list[float], list[float]]:
    lower_angles = [0.0]
    lower_regular = [0.0]
    upper_angles = [0.5 * math.pi]
    upper_regular = [RIGHT_ANGLE_QI - 2.0]
    for theta, value in zip(angles, qi):
        if theta < 0.5 * math.pi:
            lower_angles.append(theta)
            lower_regular.append(value - math.log(theta / 3.0))
        elif theta == 0.5 * math.pi:
            continue
        else:
            upper_angles.append(theta)
            upper_regular.append(value - math.pi / (math.pi - theta))
    lower_angles.append(0.5 * math.pi)
    lower_regular.append(RIGHT_ANGLE_QI - math.log((0.5 * math.pi) / 3.0))
    upper_angles.append(math.pi)
    upper_regular.append(LARGE_ANGLE_CONSTANT)
    return lower_angles, lower_regular, upper_angles, upper_regular


def _endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
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
    """Fritsch--Carlson PCHIP matching the dependency-free C evaluator."""

    widths = np.diff(np.asarray(angles, dtype=float))
    secants = np.diff(np.asarray(values, dtype=float)) / widths
    tangents = np.zeros(len(angles))
    tangents[0] = _endpoint_slope(widths[0], widths[1], secants[0], secants[1])
    for index in range(1, len(angles) - 1):
        left, right = secants[index - 1], secants[index]
        if left != 0.0 and right != 0.0 and (left > 0.0) == (right > 0.0):
            weight1 = 2.0 * widths[index] + widths[index - 1]
            weight2 = widths[index] + 2.0 * widths[index - 1]
            tangents[index] = (weight1 + weight2) / (
                weight1 / left + weight2 / right
            )
    tangents[-1] = _endpoint_slope(
        widths[-1], widths[-2], secants[-1], secants[-2]
    )
    if small_angle:
        tangents[0] = 0.0
        tangents[1] = 3.0 * values[1] / widths[0] - 0.156 * widths[0]
    interval = int(np.searchsorted(angles, query, side="right") - 1)
    interval = max(0, min(interval, len(angles) - 2))
    local = (query - angles[interval]) / widths[interval]
    local2, local3 = local * local, local * local * local
    return float(
        (2.0 * local3 - 3.0 * local2 + 1.0) * values[interval]
        + (local3 - 2.0 * local2 + local)
        * widths[interval]
        * tangents[interval]
        + (-2.0 * local3 + 3.0 * local2) * values[interval + 1]
        + (local3 - local2) * widths[interval] * tangents[interval + 1]
    )


def interpolate_qi(angles: Sequence[float], qi: Sequence[float], theta: float) -> float:
    lower_a, lower_r, upper_a, upper_r = _endpoint_augmented(angles, qi)
    if theta <= 0.5 * math.pi:
        return math.log(theta / 3.0) + pchip(
            lower_a, lower_r, theta, small_angle=True
        )
    return math.pi / (math.pi - theta) + pchip(upper_a, upper_r, theta)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_array(name: str, values: Sequence[float]) -> str:
    lines = [f"static const double {name}[{len(values)}] = {{"]
    for start in range(0, len(values), 4):
        chunk = ", ".join(f"{value:.17g}" for value in values[start : start + 4])
        suffix = "," if start + 4 < len(values) else ""
        lines.append(f"  {chunk}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def _write_header(
    path: Path,
    angles: Sequence[float],
    qi: Sequence[float],
    *,
    source_sha: str,
    table_sha: str,
    checkpoint_sha: str,
) -> None:
    lower_a, lower_r, upper_a, upper_r = _endpoint_augmented(angles, qi)
    text = f"""/**
 * Generated Scott--Hocking one-phase reference data.  Do not edit.
 *
 * Generator: gle-ode/reference-generator/scott_hocking.py
 * Method: {METHOD}
 * Generator SHA-256: {source_sha}
 * Node CSV SHA-256: {table_sha}
 * Checkpoint CSV SHA-256: {checkpoint_sha}
 */
#ifndef GLE_SLIP_SCOTT_DATA_H
#define GLE_SLIP_SCOTT_DATA_H

#define GLE_SCOTT_LOWER_COUNT {len(lower_a)}
#define GLE_SCOTT_UPPER_COUNT {len(upper_a)}

{_format_array("gle_scott_lower_theta", lower_a)}

{_format_array("gle_scott_lower_regular", lower_r)}

{_format_array("gle_scott_upper_theta", upper_a)}

{_format_array("gle_scott_upper_regular", upper_r)}

#endif
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_published() -> dict[float, float]:
    with PUBLISHED_TABLE.open(newline="", encoding="utf-8") as stream:
        return {
            float(row["theta_rad"]): float(row["Qi_scott"])
            for row in csv.DictReader(stream)
        }


def generate(args: argparse.Namespace) -> None:
    source_sha = _source_sha256()
    angles = default_node_angles()
    node_results: list[ConvergedResult] = []
    for index, theta in enumerate(angles, start=1):
        if theta == 0.5 * math.pi:
            primary = SolveResult(theta, RIGHT_ANGLE_QI, 0.0, 0.0, 0, 0.0)
            result = ConvergedResult(
                primary, RIGHT_ANGLE_QI, RIGHT_ANGLE_QI, 0.0, 0.0, 0.0
            )
        else:
            result = solve_converged(theta, series_terms=args.series_terms)
        node_results.append(result)
        print(
            f"node {index:02d}/{len(angles)} theta={theta:.12g} "
            f"Qi={result.primary.qi:.12g} err={result.estimated_error_qi:.3e}",
            file=sys.stderr,
        )
    qi = [result.primary.qi for result in node_results]
    rows = []
    for result in node_results:
        theta = result.primary.theta_rad
        log_c = math.log(math.sin(theta)) - result.primary.qi
        rows.append(
            {
                "theta_rad": f"{theta:.17g}",
                "Qi": f"{result.primary.qi:.17g}",
                "Q_chan": f"{1.0 + result.primary.qi:.17g}",
                "log_c": f"{log_c:.17g}",
                "c": f"{math.exp(log_c):.17g}",
                "spacing": f"{result.primary.spacing:.17g}",
                "radius": f"{result.primary.radius:.17g}",
                "series_terms": str(result.primary.series_terms),
                "coarse_Qi": f"{result.coarse_qi:.17g}",
                "wide_Qi": f"{result.wide_qi:.17g}",
                "discretisation_delta_Qi": f"{result.discretisation_delta_qi:.17g}",
                "domain_delta_Qi": f"{result.domain_delta_qi:.17g}",
                "estimated_error_Qi": f"{result.estimated_error_qi:.17g}",
                "relative_residual": f"{result.primary.relative_residual:.17g}",
                "method": METHOD,
                "generator_sha256": source_sha,
            }
        )
    fields = list(rows[0])
    _write_csv(args.table, fields, rows)

    checkpoint_rows = []
    numerical_midpoints = [0.5 * angles[0]]
    numerical_midpoints.extend(
        0.5 * (left + right)
        for left, right in zip(angles[:-1], angles[1:])
    )
    numerical_midpoints.append(0.5 * (angles[-1] + math.pi))
    for index, theta in enumerate(numerical_midpoints, start=1):
        direct = solve_converged(theta, series_terms=args.series_terms)
        interpolated = interpolate_qi(angles, qi, theta)
        discrepancy = abs(interpolated - direct.primary.qi)
        checkpoint_rows.append(
            {
                "theta_rad": f"{theta:.17g}",
                "Qi_direct": f"{direct.primary.qi:.17g}",
                "Qi_interpolated": f"{interpolated:.17g}",
                "abs_interpolation_error_Qi": f"{discrepancy:.17g}",
                "estimated_error_Qi": f"{direct.estimated_error_qi:.17g}",
                "spacing": f"{direct.primary.spacing:.17g}",
                "radius": f"{direct.primary.radius:.17g}",
                "relative_residual": f"{direct.primary.relative_residual:.17g}",
                "method": METHOD,
                "generator_sha256": source_sha,
            }
        )
        print(
            f"checkpoint {index:02d}/{len(numerical_midpoints)} "
            f"theta={theta:.12g} interpolation={discrepancy:.3e}",
            file=sys.stderr,
        )
    _write_csv(args.checkpoints, list(checkpoint_rows[0]), checkpoint_rows)

    table_sha = _file_sha256(args.table)
    checkpoint_sha = _file_sha256(args.checkpoints)
    published = _read_published()
    published_errors = []
    for theta, reference in published.items():
        published_errors.append(abs(interpolate_qi(angles, qi, theta) - reference))
    exact_error = abs(interpolate_qi(angles, qi, 0.5 * math.pi) - RIGHT_ANGLE_QI)
    max_node_error = max(result.estimated_error_qi for result in node_results)
    max_checkpoint_error = max(
        float(row["abs_interpolation_error_Qi"]) for row in checkpoint_rows
    )
    max_checkpoint_budget = max(
        float(row["abs_interpolation_error_Qi"])
        + float(row["estimated_error_Qi"])
        for row in checkpoint_rows
    )
    manifest = {
        "schema_version": 1,
        "method": METHOD,
        "authority": {
            "citation": "Julian F. Scott, QJMAM 73 (2020) 279-291",
            "doi": "10.1093/qjmam/hbaa012",
            "open_manuscript": "https://hal.science/hal-03227614v1",
            "equations": ["2.1", "2.2", "2.3", "2.11", "B.4", "B.5"],
        },
        "generator_sha256": source_sha,
        "node_csv": os.path.relpath(args.table, REPOSITORY),
        "node_csv_sha256": table_sha,
        "checkpoint_csv": os.path.relpath(args.checkpoints, REPOSITORY),
        "checkpoint_csv_sha256": checkpoint_sha,
        "node_count": len(rows),
        "checkpoint_count": len(checkpoint_rows),
        "theta_min": angles[0],
        "theta_max": angles[-1],
        "max_node_sensitivity_Qi": max_node_error,
        "max_midpoint_interpolation_error_Qi": max_checkpoint_error,
        "max_midpoint_error_budget_Qi": max_checkpoint_budget,
        "max_published_table_error_Qi": max(published_errors),
        "right_angle_analytic_error_Qi": exact_error,
        "tolerances": {
            "node_sensitivity_Qi": args.tolerance_node,
            "midpoint_interpolation_Qi": args.tolerance_interpolation,
            "midpoint_error_budget_Qi": args.tolerance_budget,
            "published_table_Qi": args.tolerance_published,
            "linear_residual": args.tolerance_residual,
        },
        "endpoint_policy": {
            "small_angle": "Qi=log(theta/3)+regular PCHIP; leading regular coefficient 0.156 enforced",
            "large_angle": "Qi=pi/(pi-theta)+regular PCHIP; endpoint constant gamma-log(2)-2",
            "note": "endpoint asymptotic intervals are not labelled analytic exact",
        },
    }
    failures = []
    if max_node_error > args.tolerance_node:
        failures.append("node sensitivity")
    if max_checkpoint_error > args.tolerance_interpolation:
        failures.append("midpoint interpolation")
    if max_checkpoint_budget > args.tolerance_budget:
        failures.append("midpoint error budget")
    if max(published_errors) > args.tolerance_published:
        failures.append("published table")
    if max(result.primary.relative_residual for result in node_results) > args.tolerance_residual:
        failures.append("linear residual")
    manifest["converged"] = not failures
    manifest["failures"] = failures
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Scott--Hocking freeze failed: " + ", ".join(failures))
    _write_header(
        args.header,
        angles,
        qi,
        source_sha=source_sha,
        table_sha=table_sha,
        checkpoint_sha=checkpoint_sha,
    )
    print(json.dumps(manifest, indent=2))


def _load_nodes(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [float(row["theta_rad"]) for row in rows], [float(row["Qi"]) for row in rows]


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = {
        "generator_sha256": _source_sha256(),
        "node_csv_sha256": _file_sha256(args.table),
        "checkpoint_csv_sha256": _file_sha256(args.checkpoints),
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise RuntimeError(f"stale Scott--Hocking evidence: {key}")
    if not manifest.get("converged"):
        raise RuntimeError("Scott--Hocking manifest is not converged")
    angles, qi = _load_nodes(args.table)
    with args.checkpoints.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            theta = float(row["theta_rad"])
            recorded = float(row["Qi_interpolated"])
            actual = interpolate_qi(angles, qi, theta)
            if not math.isclose(recorded, actual, rel_tol=0.0, abs_tol=5.0e-14):
                raise RuntimeError(f"stale midpoint interpolation at theta={theta}")
    print(
        f"verified {manifest['node_count']} Scott--Hocking nodes and "
        f"{manifest['checkpoint_count']} independent midpoints"
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subparsers = command.add_subparsers(dest="command", required=True)
    solve_parser = subparsers.add_parser("solve", help="solve one angle")
    solve_parser.add_argument("--theta-rad", type=float, required=True)
    solve_parser.add_argument("--spacing", type=float, default=0.05)
    solve_parser.add_argument("--radius", type=float, default=40.0)
    solve_parser.add_argument("--series-terms", type=int, default=3000)

    generate_parser = subparsers.add_parser("generate", help="freeze table and evidence")
    generate_parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    generate_parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    generate_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    generate_parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    generate_parser.add_argument("--series-terms", type=int, default=3000)
    generate_parser.add_argument("--tolerance-node", type=float, default=8.0e-5)
    generate_parser.add_argument(
        "--tolerance-interpolation", type=float, default=8.0e-5
    )
    generate_parser.add_argument("--tolerance-budget", type=float, default=2.5e-4)
    generate_parser.add_argument("--tolerance-published", type=float, default=3.0e-5)
    generate_parser.add_argument("--tolerance-residual", type=float, default=2.0e-13)

    verify_parser = subparsers.add_parser("verify", help="verify frozen hashes and interpolation")
    verify_parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    verify_parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    verify_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "solve":
        print(json.dumps(asdict(solve_qi(
            args.theta_rad,
            spacing=args.spacing,
            radius=args.radius,
            series_terms=args.series_terms,
        )), indent=2))
    elif args.command == "generate":
        generate(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
