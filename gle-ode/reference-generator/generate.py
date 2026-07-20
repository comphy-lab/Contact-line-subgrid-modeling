#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#   "numpy==2.0.2",
#   "scipy==1.13.1",
#   "scikit-fem==11.0.0",
# ]
# ///
"""Generate the constant-slip two-phase wedge reference for Q(theta, M).

The computation is deliberately independent of the C implementation.  It
solves the two Stokes sectors with Taylor--Hood elements, enforces the
kinematic interface conditions with Lagrange multipliers, and obtains the
finite force constant from the integrated Navier wall traction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from skfem import Basis, BilinearForm, FacetBasis, LinearForm, MeshTri, asm
from skfem.element import ElementTriP1, ElementTriP2, ElementVector
from skfem.helpers import ddot, div, sym_grad


GENERATOR_VERSION = "2"
GENERATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EULER_GAMMA = 0.5772156649015328606


@dataclass(frozen=True)
class MeshSpec:
    """Nondimensional annular-sector discretisation; the slip length is one."""

    inner_radius: float = 1.0e-5
    outer_radius: float = 1.0e6
    radial_cells: int = 320
    angular_cells_a: Optional[int] = None
    angular_cells_b: Optional[int] = None

    def validate(self) -> None:
        if not 0.0 < self.inner_radius < 1.0:
            raise ValueError("inner_radius must lie between zero and one")
        if not self.outer_radius > 1.0:
            raise ValueError("outer_radius must exceed one")
        if self.radial_cells < 8:
            raise ValueError("radial_cells must be at least eight")


@dataclass(frozen=True)
class FitWindow:
    """Radial interval used to isolate the logarithmic force plateau."""

    lower: float
    upper: float

    def validate(self, mesh: MeshSpec) -> None:
        if not mesh.inner_radius < self.lower < self.upper < mesh.outer_radius:
            raise ValueError("fit window must lie strictly inside the annulus")
        if math.log(self.upper / self.lower) < math.log(5.0):
            raise ValueError("fit window must span at least ln(5) in radius")


@dataclass
class PhaseSystem:
    """Finite-element objects and boundary labels for one wedge sector."""

    name: str
    viscosity: float
    mesh: MeshTri
    velocity_basis: Basis
    pressure_basis: Basis
    matrix: sp.csr_matrix
    rhs: np.ndarray
    known: Dict[int, float]
    wall_facets: np.ndarray
    interface_facets: np.ndarray
    offset: int = 0

    @property
    def velocity_size(self) -> int:
        return self.velocity_basis.N

    @property
    def pressure_size(self) -> int:
        return self.pressure_basis.N

    @property
    def size(self) -> int:
        return self.velocity_size + self.pressure_size


@dataclass(frozen=True)
class WedgeResult:
    """One numerical extraction of the matching constant."""

    theta_rad: float
    theta_deg: float
    viscosity_ratio: float
    Q: float
    log_c: float
    c: float
    force_intercept: float
    slope_expected: float
    slope_fitted: float
    slope_relative_error: float
    plateau_standard_deviation: float
    plateau_half_drift: float
    fit_lower: float
    fit_upper: float
    inner_radius: float
    outer_radius: float
    radial_cells: int
    angular_cells_a: int
    angular_cells_b: int
    velocity_dofs: int
    pressure_dofs: int
    interface_constraints: int
    linear_system_backward_error: float
    constraint_relative_error: float
    generator_version: str = GENERATOR_VERSION
    generator_source_sha256: str = GENERATOR_SOURCE_SHA256


@BilinearForm
def _viscous(u, v, w):
    return 2.0 * w.mu * ddot(sym_grad(u), sym_grad(v))


@BilinearForm
def _wall_robin(u, v, w):
    return w.mu * u[0] * v[0]


@LinearForm
def _wall_velocity(v, w):
    return w.mu * v[0]


@LinearForm
def _unit_pressure(q, _w):
    return q


@BilinearForm
def _velocity_divergence(u, q, _w):
    return q * div(u)


def cox_mobility(theta: float, viscosity_ratio: float) -> float:
    """Return Chan et al.'s signed Cox mobility F(theta, M)."""

    if not 0.0 < theta < math.pi:
        raise ValueError("theta must lie strictly between zero and pi")
    if viscosity_ratio < 0.0:
        raise ValueError("viscosity_ratio must be non-negative")

    def f1(angle: float) -> float:
        return angle * angle - math.sin(angle) ** 2

    def f2(angle: float) -> float:
        return angle - math.sin(angle) * math.cos(angle)

    def f3(angle: float) -> float:
        return angle * (math.pi - angle) + math.sin(angle) ** 2

    complement = math.pi - theta
    numerator = (
        viscosity_ratio**2 * f1(theta)
        + 2.0 * viscosity_ratio * f3(theta)
        + f1(complement)
    )
    denominator = (
        viscosity_ratio * f1(theta) * f2(complement)
        + f1(complement) * f2(theta)
    )
    return -(2.0 * math.sin(theta) ** 3 / 3.0) * numerator / denominator


def scott_hocking_right_angle_Q() -> float:
    """Exact single-phase value Q(pi/2, 0)."""

    return 1.0 + EULER_GAMMA - math.log(2.0)


def corrected_right_angle_Q(viscosity_ratio: float) -> float:
    """Corrected two-phase right-angle expression quoted by Luo and Gao."""

    if viscosity_ratio < 0.0:
        raise ValueError("viscosity_ratio must be non-negative")
    h_a = 4.0 * (EULER_GAMMA - math.log(2.0)) / math.pi
    h_b = -1.539
    m = viscosity_ratio
    numerator = math.pi * (math.pi**2 - 4.0) * (
        (m - 1.0) ** 2 * h_a + 4.0 * m * h_b
    )
    denominator = 4.0 * math.pi**2 * (m + 1.0) ** 2 - 16.0 * (m - 1.0) ** 2
    return 1.0 + numerator / denominator


def _streamfunction_rows(angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows mapping biharmonic wedge coefficients to f, f', and f''."""

    sine = math.sin(angle)
    cosine = math.cos(angle)
    value = np.array(
        [sine, cosine, angle * sine, angle * cosine], dtype=float
    )
    first = np.array(
        [cosine, -sine, sine + angle * cosine, cosine - angle * sine],
        dtype=float,
    )
    second = np.array(
        [-sine, -cosine, 2.0 * cosine - angle * sine,
         -2.0 * sine - angle * cosine],
        dtype=float,
    )
    return value, first, second


def huh_scriven_coefficients(
    theta: float, viscosity_ratio: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Solve for the leading no-slip wedge velocity in each phase."""

    zero_value, zero_first, _ = _streamfunction_rows(0.0)
    theta_value, theta_first, theta_second = _streamfunction_rows(theta)

    if viscosity_ratio == 0.0:
        matrix = np.vstack((zero_value, zero_first, theta_value, theta_second))
        rhs = np.array([0.0, 1.0, 0.0, 0.0])
        return np.linalg.solve(matrix, rhs), None

    pi_value, pi_first, _ = _streamfunction_rows(math.pi)
    matrix = np.zeros((8, 8), dtype=float)
    rhs = np.zeros(8, dtype=float)
    matrix[0, :4] = zero_value
    matrix[1, :4] = zero_first
    rhs[1] = 1.0
    matrix[2, 4:] = pi_value
    matrix[3, 4:] = pi_first
    rhs[3] = -1.0
    matrix[4, :4] = theta_value
    matrix[5, 4:] = theta_value
    matrix[6, :4] = theta_first
    matrix[6, 4:] = -theta_first
    matrix[7, :4] = theta_second
    matrix[7, 4:] = -viscosity_ratio * theta_second
    coefficients = np.linalg.solve(matrix, rhs)
    return coefficients[:4], coefficients[4:]


def _outer_velocity(angle: float, coefficients: np.ndarray) -> np.ndarray:
    value, first, _ = _streamfunction_rows(angle)
    radial = float(first @ coefficients)
    azimuthal = -float(value @ coefficients)
    return np.array(
        [
            radial * math.cos(angle) - azimuthal * math.sin(angle),
            radial * math.sin(angle) + azimuthal * math.cos(angle),
        ]
    )


def _sector_mesh(
    angle_start: float,
    angle_end: float,
    spec: MeshSpec,
    angular_cells: Optional[int],
) -> Tuple[MeshTri, int]:
    log_step = math.log(spec.outer_radius / spec.inner_radius) / spec.radial_cells
    if angular_cells is None:
        angular_cells = max(2, int(math.ceil((angle_end - angle_start) / log_step)))
    if angular_cells < 2:
        raise ValueError("each sector needs at least two angular cells")

    radii = np.geomspace(
        spec.inner_radius, spec.outer_radius, spec.radial_cells + 1
    )
    angles = np.linspace(angle_start, angle_end, angular_cells + 1)
    points = np.empty((2, (spec.radial_cells + 1) * (angular_cells + 1)))

    def vertex(radial_index: int, angular_index: int) -> int:
        return radial_index * (angular_cells + 1) + angular_index

    for radial_index, radius in enumerate(radii):
        sl = slice(
            radial_index * (angular_cells + 1),
            (radial_index + 1) * (angular_cells + 1),
        )
        points[0, sl] = radius * np.cos(angles)
        points[1, sl] = radius * np.sin(angles)

    triangles: List[Tuple[int, int, int]] = []
    for radial_index in range(spec.radial_cells):
        for angular_index in range(angular_cells):
            lower_inner = vertex(radial_index, angular_index)
            lower_outer = vertex(radial_index + 1, angular_index)
            upper_outer = vertex(radial_index + 1, angular_index + 1)
            upper_inner = vertex(radial_index, angular_index + 1)
            if (radial_index + angular_index) % 2 == 0:
                triangles.extend(
                    [
                        (lower_inner, lower_outer, upper_outer),
                        (lower_inner, upper_outer, upper_inner),
                    ]
                )
            else:
                triangles.extend(
                    [
                        (lower_inner, lower_outer, upper_inner),
                        (lower_outer, upper_outer, upper_inner),
                    ]
                )
    mesh = MeshTri(points, np.asarray(triangles, dtype=np.int32).T)
    return mesh, angular_cells


def _boundary_facets(
    mesh: MeshTri,
    angle_start: float,
    angle_end: float,
    spec: MeshSpec,
) -> Dict[str, np.ndarray]:
    boundary = mesh.boundary_facets()
    vertices = mesh.facets[:, boundary]
    coordinates = mesh.p[:, vertices]
    radii = np.sqrt(np.sum(coordinates**2, axis=0))
    angles = np.mod(np.arctan2(coordinates[1], coordinates[0]), 2.0 * math.pi)
    tolerance = 2.0e-10

    def angular_distance(values: np.ndarray, target: float) -> np.ndarray:
        return np.abs(np.angle(np.exp(1j * (values - target))))

    labels = {
        "inner": boundary[np.all(np.abs(radii / spec.inner_radius - 1.0) < tolerance, axis=0)],
        "outer": boundary[np.all(np.abs(radii / spec.outer_radius - 1.0) < tolerance, axis=0)],
        "start": boundary[np.all(angular_distance(angles, angle_start) < tolerance, axis=0)],
        "end": boundary[np.all(angular_distance(angles, angle_end) < tolerance, axis=0)],
    }
    expected = {
        "inner": None,
        "outer": None,
        "start": spec.radial_cells,
        "end": spec.radial_cells,
    }
    for name, count in expected.items():
        if labels[name].size == 0 or (count is not None and labels[name].size != count):
            raise RuntimeError("failed to identify the %s boundary" % name)
    return labels


def _component_map(basis: Basis) -> np.ndarray:
    components = np.full(basis.N, -1, dtype=np.int8)
    for component in range(2):
        components[basis.nodal_dofs[component]] = component
        components[basis.facet_dofs[component]] = component
    if np.any(components < 0):
        raise RuntimeError("unexpected non-nodal P2 velocity degree of freedom")
    return components


def _add_known(
    known: Dict[int, float], dof: int, value: float, description: str
) -> None:
    if dof in known and not math.isclose(known[dof], value, abs_tol=2.0e-10):
        raise RuntimeError("incompatible Dirichlet data at %s" % description)
    known[dof] = value


def _assemble_phase(
    name: str,
    angle_start: float,
    angle_end: float,
    interface_side: str,
    viscosity: float,
    coefficients: np.ndarray,
    spec: MeshSpec,
    angular_cells: Optional[int],
) -> Tuple[PhaseSystem, int]:
    mesh, used_angular_cells = _sector_mesh(
        angle_start, angle_end, spec, angular_cells
    )
    labels = _boundary_facets(mesh, angle_start, angle_end, spec)
    velocity_element = ElementVector(ElementTriP2())
    pressure_element = ElementTriP1()
    velocity_basis = Basis(mesh, velocity_element, intorder=4)
    pressure_basis = Basis(mesh, pressure_element, intorder=4)
    wall_side = "start" if name == "A" else "end"
    wall_facets = labels[wall_side]
    interface_facets = labels[interface_side]

    viscous = asm(_viscous, velocity_basis, mu=viscosity)
    wall_basis = FacetBasis(mesh, velocity_element, facets=wall_facets, intorder=4)
    viscous = viscous + asm(_wall_robin, wall_basis, mu=viscosity)
    divergence_matrix = asm(
        _velocity_divergence, velocity_basis, pressure_basis
    )
    matrix = sp.bmat(
        [
            [viscous, -divergence_matrix.T],
            [-divergence_matrix, None],
        ],
        format="csr",
    )
    rhs = np.concatenate(
        (
            asm(_wall_velocity, wall_basis, mu=viscosity),
            np.zeros(pressure_basis.N),
        )
    )

    components = _component_map(velocity_basis)
    known: Dict[int, float] = {}
    inner_dofs = velocity_basis.get_dofs(facets=labels["inner"]).all()
    for dof in inner_dofs:
        _add_known(known, int(dof), 0.0, "%s inner arc" % name)

    outer_dofs = velocity_basis.get_dofs(facets=labels["outer"]).all()
    for dof in outer_dofs:
        x, y = velocity_basis.doflocs[:, dof]
        angle = float(np.mod(math.atan2(y, x), 2.0 * math.pi))
        value = _outer_velocity(angle, coefficients)[components[dof]]
        _add_known(known, int(dof), float(value), "%s outer arc" % name)

    wall_normal_dofs = velocity_basis.get_dofs(facets=wall_facets).all(["u^2"])
    for dof in wall_normal_dofs:
        _add_known(known, int(dof), 0.0, "%s wall normal velocity" % name)

    return (
        PhaseSystem(
            name=name,
            viscosity=viscosity,
            mesh=mesh,
            velocity_basis=velocity_basis,
            pressure_basis=pressure_basis,
            matrix=matrix,
            rhs=rhs,
            known=known,
            wall_facets=wall_facets,
            interface_facets=interface_facets,
        ),
        used_angular_cells,
    )


def _trace_dofs(phase: PhaseSystem, spec: MeshSpec) -> List[Tuple[float, int, int]]:
    view = phase.velocity_basis.get_dofs(facets=phase.interface_facets)
    horizontal = view.all(["u^1"])
    vertical = view.all(["u^2"])
    horizontal = horizontal[np.argsort(np.linalg.norm(
        phase.velocity_basis.doflocs[:, horizontal], axis=0
    ))]
    vertical = vertical[np.argsort(np.linalg.norm(
        phase.velocity_basis.doflocs[:, vertical], axis=0
    ))]
    radii_x = np.linalg.norm(phase.velocity_basis.doflocs[:, horizontal], axis=0)
    radii_y = np.linalg.norm(phase.velocity_basis.doflocs[:, vertical], axis=0)
    if not np.allclose(radii_x, radii_y, rtol=2.0e-12, atol=2.0e-12):
        raise RuntimeError("interface velocity components do not share trace nodes")
    trace = []
    for radius, dof_x, dof_y in zip(radii_x, horizontal, vertical):
        if math.isclose(radius, spec.inner_radius, rel_tol=2.0e-10):
            continue
        if math.isclose(radius, spec.outer_radius, rel_tol=2.0e-10):
            continue
        trace.append((float(radius), int(dof_x), int(dof_y)))
    return trace


def _constraint_matrix(
    phases: Sequence[PhaseSystem], theta: float, spec: MeshSpec
) -> sp.csr_matrix:
    row_indices: List[int] = []
    column_indices: List[int] = []
    values: List[float] = []
    row = 0

    def add(entries: Iterable[Tuple[int, float]]) -> None:
        nonlocal row
        for column, value in entries:
            row_indices.append(row)
            column_indices.append(column)
            values.append(value)
        row += 1

    normal = np.array([-math.sin(theta), math.cos(theta)])
    tangent = np.array([math.cos(theta), math.sin(theta)])
    trace_a = _trace_dofs(phases[0], spec)
    if len(phases) == 1:
        for _, dof_x, dof_y in trace_a:
            add(
                (
                    (phases[0].offset + dof_x, normal[0]),
                    (phases[0].offset + dof_y, normal[1]),
                )
            )
    else:
        trace_b = _trace_dofs(phases[1], spec)
        if len(trace_a) != len(trace_b):
            raise RuntimeError("the two interface traces have different sizes")
        for a, b in zip(trace_a, trace_b):
            if not math.isclose(a[0], b[0], rel_tol=2.0e-10, abs_tol=2.0e-12):
                raise RuntimeError("the two interface trace nodes do not coincide")
            _, ax, ay = a
            _, bx, by = b
            oa = phases[0].offset
            ob = phases[1].offset
            add(((oa + ax, normal[0]), (oa + ay, normal[1])))
            add(((ob + bx, normal[0]), (ob + by, normal[1])))
            add(
                (
                    (oa + ax, tangent[0]),
                    (oa + ay, tangent[1]),
                    (ob + bx, -tangent[0]),
                    (ob + by, -tangent[1]),
                )
            )

    for phase in phases:
        pressure_weights = asm(_unit_pressure, phase.pressure_basis)
        pressure_columns = (
            phase.offset + phase.velocity_size + np.arange(phase.pressure_size)
        )
        add(zip(pressure_columns.tolist(), pressure_weights.tolist()))

    matrix = sp.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(row, sum(phase.size for phase in phases)),
    ).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(norms == 0.0):
        raise RuntimeError("empty interface or pressure constraint")
    return sp.diags(1.0 / norms) @ matrix


def _solve_kkt(
    phases: Sequence[PhaseSystem], constraints: sp.csr_matrix
) -> Tuple[List[np.ndarray], float, float]:
    matrix = sp.block_diag([phase.matrix for phase in phases], format="csr")
    rhs = np.concatenate([phase.rhs for phase in phases])
    total_size = matrix.shape[0]

    known: Dict[int, float] = {}
    scale = np.ones(total_size)
    for phase in phases:
        for local_dof, value in phase.known.items():
            known[phase.offset + local_dof] = value
        pressure_dofs = (
            phase.offset
            + phase.velocity_size
            + np.arange(phase.pressure_size)
        )
        pressure_radii = np.linalg.norm(phase.pressure_basis.doflocs, axis=0)
        scale[pressure_dofs] = 1.0 / pressure_radii

    scaling = sp.diags(scale)
    matrix = scaling @ matrix @ scaling
    rhs = scale * rhs
    constraints = constraints @ scaling

    fixed = np.array(sorted(known), dtype=np.int64)
    fixed_values = np.array([known[index] / scale[index] for index in fixed])
    free_mask = np.ones(total_size, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    reduced_matrix = matrix[free][:, free]
    reduced_constraints = constraints[:, free]
    reduced_rhs = rhs[free] - matrix[free][:, fixed] @ fixed_values
    constraint_rhs = -(constraints[:, fixed] @ fixed_values)
    zero = sp.csr_matrix((constraints.shape[0], constraints.shape[0]))
    kkt = sp.bmat(
        [[reduced_matrix, reduced_constraints.T], [reduced_constraints, zero]],
        format="csr",
    )
    combined_rhs = np.concatenate((reduced_rhs, np.asarray(constraint_rhs).ravel()))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        solution = spsolve(kkt, combined_rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("the Stokes KKT solve returned a non-finite value")

    algebraic_residual = np.asarray(kkt @ solution - combined_rhs).ravel()
    matrix_norm = float(np.max(np.asarray(abs(kkt).sum(axis=1)).ravel()))
    solution_norm = float(np.linalg.norm(solution, ord=np.inf))
    rhs_norm = float(np.linalg.norm(combined_rhs, ord=np.inf))
    backward_denominator = matrix_norm * solution_norm + rhs_norm
    linear_system_backward_error = float(
        np.linalg.norm(algebraic_residual, ord=np.inf)
        / max(backward_denominator, np.finfo(float).tiny)
    )

    transformed = np.zeros(total_size)
    transformed[fixed] = fixed_values
    transformed[free] = solution[: free.size]
    constraint_residual = np.asarray(constraints @ transformed).ravel()
    constraint_norm = float(
        np.max(np.asarray(abs(constraints).sum(axis=1)).ravel())
    )
    transformed_norm = float(np.linalg.norm(transformed, ord=np.inf))
    constraint_relative_error = float(
        np.linalg.norm(constraint_residual, ord=np.inf)
        / max(constraint_norm * transformed_norm, np.finfo(float).tiny)
    )
    physical = scale * transformed
    phase_solutions = [
        physical[phase.offset : phase.offset + phase.size] for phase in phases
    ]
    return (
        phase_solutions,
        linear_system_backward_error,
        constraint_relative_error,
    )


def _wall_force_increments(
    phase: PhaseSystem, solution: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    velocity = solution[: phase.velocity_size]
    contributions: List[Tuple[float, float]] = []
    for facet in phase.wall_facets:
        vertices = phase.mesh.facets[:, facet]
        endpoint_radii = np.linalg.norm(phase.mesh.p[:, vertices], axis=0)
        endpoint_dofs = phase.velocity_basis.nodal_dofs[0, vertices]
        midpoint_dof = phase.velocity_basis.facet_dofs[0, facet]
        endpoint_values = 1.0 - velocity[endpoint_dofs]
        midpoint_value = 1.0 - velocity[midpoint_dof]
        length = float(np.linalg.norm(
            phase.mesh.p[:, vertices[1]] - phase.mesh.p[:, vertices[0]]
        ))
        increment = phase.viscosity * length * (
            endpoint_values[0] + 4.0 * midpoint_value + endpoint_values[1]
        ) / 6.0
        contributions.append((float(np.max(endpoint_radii)), float(increment)))
    contributions.sort()
    return (
        np.asarray([item[0] for item in contributions]),
        np.asarray([item[1] for item in contributions]),
    )


def _combine_wall_forces(
    phases: Sequence[PhaseSystem], solutions: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    radius, increments = _wall_force_increments(phases[0], solutions[0])
    for phase, solution in zip(phases[1:], solutions[1:]):
        other_radius, other_increments = _wall_force_increments(phase, solution)
        if not np.allclose(radius, other_radius, rtol=2.0e-12, atol=2.0e-12):
            raise RuntimeError("the two wall meshes do not share radial knots")
        increments = increments + other_increments
    return radius, np.cumsum(increments)


def _default_fit_window(spec: MeshSpec) -> FitWindow:
    lower = max(8.0, min(1000.0, spec.outer_radius / 50.0))
    upper = min(spec.outer_radius / 10.0, spec.outer_radius - 1.0)
    return FitWindow(lower, upper)


def solve_wedge(
    theta: float,
    viscosity_ratio: float,
    spec: MeshSpec,
    fit_window: Optional[FitWindow] = None,
) -> WedgeResult:
    """Solve one wedge and extract Q from the force plateau."""

    spec.validate()
    if not math.radians(3.0) <= theta <= math.radians(177.0):
        raise ValueError("the FEM generator is restricted to 3 <= theta_deg <= 177")
    if viscosity_ratio < 0.0:
        raise ValueError("viscosity_ratio must be non-negative")
    if fit_window is None:
        fit_window = _default_fit_window(spec)
    fit_window.validate(spec)

    coefficients_a, coefficients_b = huh_scriven_coefficients(
        theta, viscosity_ratio
    )
    phase_a, angular_a = _assemble_phase(
        "A",
        0.0,
        theta,
        "end",
        1.0,
        coefficients_a,
        spec,
        spec.angular_cells_a,
    )
    phases = [phase_a]
    angular_b = 0
    if viscosity_ratio > 0.0:
        if coefficients_b is None:
            raise RuntimeError("missing outer wedge coefficients for phase B")
        phase_b, angular_b = _assemble_phase(
            "B",
            theta,
            math.pi,
            "start",
            viscosity_ratio,
            coefficients_b,
            spec,
            spec.angular_cells_b,
        )
        phases.append(phase_b)

    offset = 0
    for phase in phases:
        phase.offset = offset
        offset += phase.size
    constraints = _constraint_matrix(phases, theta, spec)
    solutions, linear_error, constraint_error = _solve_kkt(phases, constraints)
    radii, force = _combine_wall_forces(phases, solutions)

    mask = (radii >= fit_window.lower) & (radii <= fit_window.upper)
    if np.count_nonzero(mask) < 6:
        raise RuntimeError("fewer than six radial samples lie in the fit window")
    log_radius = np.log(radii[mask])
    plateau_force = force[mask]
    mobility = cox_mobility(theta, viscosity_ratio)
    expected_slope = -3.0 * mobility / math.sin(theta)
    fitted_slope, fitted_intercept = np.polyfit(log_radius, plateau_force, 1)
    fixed_intercepts = plateau_force - expected_slope * log_radius
    intercept = float(np.mean(fixed_intercepts))
    quarter = max(2, fixed_intercepts.size // 4)
    plateau_half_drift = float(
        abs(np.mean(fixed_intercepts[-quarter:]) - np.mean(fixed_intercepts[:quarter]))
    )
    Q = 1.0 - math.sin(theta) * intercept / (3.0 * mobility)
    log_c = math.log(math.sin(theta)) + math.sin(theta) * intercept / (3.0 * mobility)

    return WedgeResult(
        theta_rad=theta,
        theta_deg=math.degrees(theta),
        viscosity_ratio=viscosity_ratio,
        Q=float(Q),
        log_c=float(log_c),
        c=float(math.exp(log_c)),
        force_intercept=intercept,
        slope_expected=float(expected_slope),
        slope_fitted=float(fitted_slope),
        slope_relative_error=float(abs(fitted_slope / expected_slope - 1.0)),
        plateau_standard_deviation=float(np.std(fixed_intercepts, ddof=1)),
        plateau_half_drift=plateau_half_drift,
        fit_lower=fit_window.lower,
        fit_upper=fit_window.upper,
        inner_radius=spec.inner_radius,
        outer_radius=spec.outer_radius,
        radial_cells=spec.radial_cells,
        angular_cells_a=angular_a,
        angular_cells_b=angular_b,
        velocity_dofs=sum(phase.velocity_size for phase in phases),
        pressure_dofs=sum(phase.pressure_size for phase in phases),
        interface_constraints=constraints.shape[0] - len(phases),
        linear_system_backward_error=linear_error,
        constraint_relative_error=constraint_error,
    )


def _result_dict(result: WedgeResult) -> Dict[str, object]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in asdict(result).items()
    }


def _print_result(result: WedgeResult) -> None:
    print(
        "theta={:.6g} deg  M={:.6g}  Q={:.10g}  c={:.10g}".format(
            result.theta_deg, result.viscosity_ratio, result.Q, result.c
        )
    )
    print(
        "slope={:.10g} (expected {:.10g}, relative error {:.3g})".format(
            result.slope_fitted,
            result.slope_expected,
            result.slope_relative_error,
        )
    )
    print(
        "plateau sigma={:.3g}, half-drift={:.3g}, dofs=({}, {})".format(
            result.plateau_standard_deviation,
            result.plateau_half_drift,
            result.velocity_dofs,
            result.pressure_dofs,
        )
    )
    print(
        "KKT backward error={:.3g}, constraint error={:.3g}".format(
            result.linear_system_backward_error,
            result.constraint_relative_error,
        )
    )


def _mesh_spec_from_args(args: argparse.Namespace) -> MeshSpec:
    return MeshSpec(
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        radial_cells=args.radial_cells,
        angular_cells_a=args.angular_cells_a,
        angular_cells_b=args.angular_cells_b,
    )


def _fit_from_args(args: argparse.Namespace, spec: MeshSpec) -> FitWindow:
    default = _default_fit_window(spec)
    return FitWindow(
        args.fit_lower if args.fit_lower is not None else default.lower,
        args.fit_upper if args.fit_upper is not None else default.upper,
    )


def _add_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inner-radius", type=float, default=1.0e-5)
    parser.add_argument("--outer-radius", type=float, default=1.0e6)
    parser.add_argument("--radial-cells", type=int, default=320)
    parser.add_argument("--angular-cells-a", type=int)
    parser.add_argument("--angular-cells-b", type=int)
    parser.add_argument("--fit-lower", type=float)
    parser.add_argument("--fit-upper", type=float)


def _solve_command(args: argparse.Namespace) -> int:
    spec = _mesh_spec_from_args(args)
    result = solve_wedge(
        math.radians(args.theta_deg),
        args.viscosity_ratio,
        spec,
        _fit_from_args(args, spec),
    )
    if args.json:
        print(json.dumps(_result_dict(result), indent=2, sort_keys=True))
    else:
        _print_result(result)
    return 0


def _anchors_command(args: argparse.Namespace) -> int:
    spec = _mesh_spec_from_args(args)
    fit = _fit_from_args(args, spec)
    cases = [
        (0.0, scott_hocking_right_angle_Q(), "Scott--Hocking M=0"),
        (1.0, corrected_right_angle_Q(1.0), "corrected right angle M=1"),
        (10.0, corrected_right_angle_Q(10.0), "corrected right angle M=10"),
    ]
    records = []
    passed = True
    for viscosity_ratio, expected, label in cases:
        result = solve_wedge(math.pi / 2.0, viscosity_ratio, spec, fit)
        error = abs(result.Q - expected)
        case_passed = error <= args.tolerance
        passed = passed and case_passed
        records.append(
            {
                "label": label,
                "expected_Q": expected,
                "absolute_error_Q": error,
                "passed": case_passed,
                "result": _result_dict(result),
            }
        )
        if not args.json:
            _print_result(result)
            print(
                "anchor Q={:.10g}, |Delta Q|={:.3g}: {}\n".format(
                    expected, error, "PASS" if case_passed else "FAIL"
                )
            )
    if args.json:
        print(json.dumps({"passed": passed, "anchors": records}, indent=2))
    return 0 if passed else 1


def _parse_number_grid(expression: str) -> List[float]:
    """Parse comma-separated values or inclusive start:stop:step ranges."""

    values: List[float] = []
    for item in expression.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            values.append(float(item))
            continue
        start, stop, step = (float(part) for part in item.split(":"))
        if step == 0.0 or (stop - start) * step < 0.0:
            raise ValueError("invalid numeric range: %s" % item)
        count = int(math.floor((stop - start) / step + 1.0e-10)) + 1
        values.extend(start + index * step for index in range(count))
    if not values:
        raise ValueError("empty numeric grid")
    return values


def _converged_case(
    theta: float,
    viscosity_ratio: float,
    spec: MeshSpec,
    fit: FitWindow,
    tolerance_Q: float,
    tolerance_slope: float,
    tolerance_linear: float,
) -> Tuple[bool, WedgeResult, Dict[str, float]]:
    fine = solve_wedge(theta, viscosity_ratio, spec, fit)
    coarse_cells = max(8, int(round(0.75 * spec.radial_cells)))
    coarse = solve_wedge(
        theta,
        viscosity_ratio,
        MeshSpec(
            spec.inner_radius,
            spec.outer_radius,
            coarse_cells,
            spec.angular_cells_a,
            spec.angular_cells_b,
        ),
        fit,
    )
    log_span = math.log(spec.outer_radius / spec.inner_radius)
    inner_radius = spec.inner_radius / 4.0
    inner_cells = int(round(
        spec.radial_cells
        * math.log(spec.outer_radius / inner_radius)
        / log_span
    ))
    inner = solve_wedge(
        theta,
        viscosity_ratio,
        MeshSpec(inner_radius, spec.outer_radius, inner_cells),
        fit,
    )
    outer_radius = spec.outer_radius * 4.0
    outer_cells = int(round(
        spec.radial_cells
        * math.log(outer_radius / spec.inner_radius)
        / log_span
    ))
    outer = solve_wedge(
        theta,
        viscosity_ratio,
        MeshSpec(spec.inner_radius, outer_radius, outer_cells),
        fit,
    )
    if fit.lower / 10.0 > spec.inner_radius:
        shifted_fit = FitWindow(fit.lower / 10.0, fit.upper / 10.0)
    elif fit.upper * 10.0 < spec.outer_radius:
        shifted_fit = FitWindow(fit.lower * 10.0, fit.upper * 10.0)
    else:
        raise ValueError(
            "convergence certification needs room to shift the fit window "
            "by one radial decade"
        )
    shifted = solve_wedge(theta, viscosity_ratio, spec, shifted_fit)
    errors = {
        "mesh_delta_Q": abs(fine.Q - coarse.Q),
        "inner_radius_delta_Q": abs(fine.Q - inner.Q),
        "outer_radius_delta_Q": abs(fine.Q - outer.Q),
        "fit_window_delta_Q": abs(fine.Q - shifted.Q),
        "slope_relative_error": fine.slope_relative_error,
        "plateau_half_drift": fine.plateau_half_drift,
        "plateau_half_drift_Q": abs(
            math.sin(theta)
            * fine.plateau_half_drift
            / (3.0 * cox_mobility(theta, viscosity_ratio))
        ),
        "max_linear_system_backward_error": max(
            result.linear_system_backward_error
            for result in (fine, coarse, inner, outer, shifted)
        ),
        "max_constraint_relative_error": max(
            result.constraint_relative_error
            for result in (fine, coarse, inner, outer, shifted)
        ),
    }
    errors["estimated_error_Q"] = max(
        errors["mesh_delta_Q"],
        errors["inner_radius_delta_Q"],
        errors["outer_radius_delta_Q"],
        errors["fit_window_delta_Q"],
        errors["plateau_half_drift_Q"],
    )
    passed = (
        errors["estimated_error_Q"] <= tolerance_Q
        and fine.slope_relative_error <= tolerance_slope
        and errors["plateau_half_drift_Q"] <= tolerance_Q
        and errors["max_linear_system_backward_error"] <= tolerance_linear
        and errors["max_constraint_relative_error"] <= tolerance_linear
    )
    return passed, fine, errors


def _table_command(args: argparse.Namespace) -> int:
    theta_degrees = _parse_number_grid(args.theta_deg_grid)
    log_ratios = _parse_number_grid(args.log10_M_grid)
    if any(log_ratio > 0.0 for log_ratio in log_ratios):
        raise ValueError(
            "store only log10(M) <= 0; the runtime obtains M > 1 by phase exchange"
        )
    spec = _mesh_spec_from_args(args)
    fit = _fit_from_args(args, spec)
    records = []
    failures = []
    for theta_deg in theta_degrees:
        for log_ratio in log_ratios:
            ratio = 10.0**log_ratio
            print(
                "checking theta={:.6g} deg, log10(M)={:.6g}".format(
                    theta_deg, log_ratio
                ),
                file=sys.stderr,
            )
            passed, result, errors = _converged_case(
                math.radians(theta_deg),
                ratio,
                spec,
                fit,
                args.tolerance_Q,
                args.tolerance_slope,
                args.tolerance_linear,
            )
            record = _result_dict(result)
            record.update(errors)
            record["log10_viscosity_ratio"] = log_ratio
            record["converged"] = passed
            records.append(record)
            if not passed:
                failures.append(
                    {
                        "theta_deg": theta_deg,
                        "log10_viscosity_ratio": log_ratio,
                        **errors,
                    }
                )

    if failures and not args.allow_unconverged:
        print(json.dumps({"convergence_failures": failures}, indent=2), file=sys.stderr)
        print("refusing to write a table containing unconverged rows", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        temporary = Path(stream.name)
    temporary.replace(output)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "schema": "table.schema.json",
        "table": output.name,
        "table_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "interpolated_quantity": "Q",
        "c_reconstruction": "log(c) = 1 + log(sin(theta_rad)) - Q",
        "method": "two-sector P2/P1 Stokes FEM with equal Navier slip",
        "rows": len(records),
        "all_converged": not failures,
        "theta_deg_grid": theta_degrees,
        "log10_M_grid": log_ratios,
        "mesh": asdict(spec),
        "fit_window": asdict(fit),
        "tolerance_Q": args.tolerance_Q,
        "tolerance_slope": args.tolerance_slope,
        "tolerance_linear": args.tolerance_linear,
        "convergence_checks": [
            "coarser mesh",
            "inner radius divided by four at fixed logarithmic resolution",
            "outer radius multiplied by four at fixed logarithmic resolution",
            "fit window shifted by one radial decade",
            "free fitted Cox slope",
            "within-window intercept drift",
            "KKT normwise backward error",
            "interface and pressure-gauge constraint residual",
        ],
        "references": [
            "https://doi.org/10.1017/jfm.2020.499",
            "https://doi.org/10.1017/jfm.2025.10587",
            "https://doi.org/10.1093/qjmam/hbaa012",
        ],
        "failures": failures,
    }
    manifest_path = output.with_suffix(".manifest.json")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent, delete=False
    ) as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    temporary.replace(manifest_path)
    print("wrote %s and its convergence manifest" % output)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="solve one wedge")
    solve_parser.add_argument("--theta-deg", type=float, required=True)
    solve_parser.add_argument("--viscosity-ratio", "-M", type=float, required=True)
    solve_parser.add_argument("--json", action="store_true")
    _add_mesh_arguments(solve_parser)
    solve_parser.set_defaults(function=_solve_command)

    anchors_parser = subparsers.add_parser(
        "anchors", help="check the public right-angle anchors"
    )
    anchors_parser.add_argument("--tolerance", type=float, default=2.0e-3)
    anchors_parser.add_argument("--json", action="store_true")
    _add_mesh_arguments(anchors_parser)
    anchors_parser.set_defaults(function=_anchors_command)

    table_parser = subparsers.add_parser(
        "table", help="generate only rows that pass convergence checks"
    )
    table_parser.add_argument("--theta-deg-grid", required=True)
    table_parser.add_argument("--log10-M-grid", required=True)
    table_parser.add_argument("--output", required=True)
    table_parser.add_argument("--tolerance-Q", type=float, default=1.0e-3)
    table_parser.add_argument("--tolerance-slope", type=float, default=2.0e-3)
    table_parser.add_argument("--tolerance-linear", type=float, default=1.0e-10)
    table_parser.add_argument("--allow-unconverged", action="store_true")
    _add_mesh_arguments(table_parser)
    table_parser.set_defaults(function=_table_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
