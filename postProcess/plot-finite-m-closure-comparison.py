#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Plot the finite-viscosity-ratio Luo--Gao matching error.

The numerical two-phase constant-slip Stokes-wedge table is the reference.
At each certified table node this script evaluates Luo & Gao (2025), Eq.
(4.10), and plots

    Q_LG - Q_FEM,    c_lambda,LG / c_lambda,FEM = exp(Q_FEM - Q_LG).

Only the actual, nonuniformly spaced table nodes are drawn.  There is no
interpolation, contouring, or raster resampling between them.

Usage
-----
uv run postProcess/plot-finite-m-closure-comparison.py
uv run postProcess/plot-finite-m-closure-comparison.py \
  --table gle-ode/reference-generator/data/two-phase-q.csv --output-dir img
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
if shutil.which("latex"):
    matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
else:
    matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"

import matplotlib.pyplot as plt  # noqa: E402  (backend selected above)
import numpy as np  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
DEFAULT_TABLE = REPO / "gle-ode" / "reference-generator" / "data" / "two-phase-q.csv"
DEFAULT_OUTPUT_DIR = REPO / "img"
OUTPUT_STEM = "finite-m-closure-comparison"


@dataclass(frozen=True)
class Node:
    theta_deg: float
    log10_m: float
    q_fem: float
    q_lg: float

    @property
    def delta_q(self) -> float:
        return self.q_lg - self.q_fem

    @property
    def c_ratio(self) -> float:
        return math.exp(self.q_fem - self.q_lg)


def _f1(theta: float) -> float:
    """Huh--Scriven f1, including the C runtime's small-angle series."""
    if abs(theta) < 0.02:
        theta2 = theta * theta
        return (
            theta2
            * theta2
            / 3.0
            * (1.0 - 2.0 * theta2 / 15.0 + theta2 * theta2 / 105.0)
        )
    sine = math.sin(theta)
    return theta * theta - sine * sine


def _f2(theta: float) -> float:
    """Huh--Scriven f2, including the C runtime's small-angle series."""
    if abs(theta) < 0.02:
        theta2 = theta * theta
        return (
            2.0
            * theta
            * theta2
            / 3.0
            * (1.0 - theta2 / 5.0 + 2.0 * theta2 * theta2 / 105.0)
        )
    return theta - math.sin(theta) * math.cos(theta)


def _f3(theta: float) -> float:
    sine = math.sin(theta)
    return theta * (math.pi - theta) + sine * sine


def luo_gao_q(theta: float, viscosity_ratio: float) -> float:
    """Evaluate Luo--Gao Eq. (4.10) as in ``gle-slip-closure.h``.

    The logarithmic matching term is formed from ``log(4AC/B^2)``.  This
    avoids cancellation both when the discriminant tends to zero and when its
    square root rounds to one.
    """
    if not (0.0 < theta < math.pi and viscosity_ratio > 0.0):
        raise ValueError("finite-M Luo--Gao matching requires 0 < theta < pi and M > 0")

    sine = math.sin(theta)
    sine2 = sine * sine
    complement = math.pi - theta
    f1_theta, f2_theta = _f1(theta), _f2(theta)
    f1_comp, f2_comp = _f1(complement), _f2(complement)
    m = viscosity_ratio

    a = m * f1_theta * f2_comp + f1_comp * f2_theta
    b = 2.0 * sine2 * (m * f1_theta + f1_comp) + 2.0 * (m + 1.0) * f2_theta * f2_comp
    c = 4.0 * sine2 * (m * f2_theta + f2_comp)
    d = 2.0 * sine * (m * m * f1_theta + 2.0 * m * _f3(theta) + f1_comp)
    e = 4.0 * sine * (m * m * f2_theta + m * math.pi + f2_comp)
    if not all(math.isfinite(value) and value > 0.0 for value in (a, b, c, d, e)):
        raise ArithmeticError(
            "non-positive Luo--Gao coefficient on the physical branch"
        )

    log_z = math.log(4.0) + math.log(a) + math.log(c) - 2.0 * math.log(b)
    roundoff = (
        128.0
        * sys.float_info.epsilon
        * (1.0 + abs(math.log(a)) + abs(math.log(b)) + abs(math.log(c)))
    )
    if log_z > roundoff:
        raise ArithmeticError("negative Luo--Gao discriminant")
    log_z = min(log_z, 0.0)
    x = math.sqrt(max(0.0, -math.expm1(log_z)))

    a_over_b, e_over_d = a / b, e / d
    if hasattr(math, "fma"):
        coefficient = math.fma(2.0 * a_over_b, e_over_d, -1.0)
    else:
        coefficient = 2.0 * a_over_b * e_over_d - 1.0
    if not math.isfinite(coefficient):
        log_coefficient = (
            math.log(2.0) + math.log(a) + math.log(e) - math.log(b) - math.log(d)
        )
        if log_coefficient > math.log(sys.float_info.max):
            raise ArithmeticError("overflow in Luo--Gao matching coefficient")
        coefficient = (
            -1.0
            if log_coefficient < math.log(sys.float_info.min)
            else math.expm1(log_coefficient)
        )

    if x <= 32.0 * sys.float_info.epsilon:
        x2 = x * x
        matching_term = coefficient * (1.0 + x2 / 3.0 + x2 * x2 / 5.0)
    elif x < 0.9:
        matching_term = coefficient * math.atanh(x) / x
    else:
        log_ratio = 2.0 * math.log1p(x) - log_z
        matching_term = coefficient * log_ratio / (2.0 * x)

    q = 1.0 + 0.5 * (math.log(a) - math.log(c)) + matching_term
    if not math.isfinite(q):
        raise ArithmeticError("non-finite Luo--Gao Q")
    return q


def load_nodes(path: Path) -> list[Node]:
    """Read and validate the frozen FEM table, then evaluate Luo--Gao Q."""
    rows: list[Node] = []
    seen: set[tuple[float, float]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            theta_deg = float(raw["theta_deg"])
            log10_m = float(raw["log10_viscosity_ratio"])
            m = float(raw["viscosity_ratio"])
            q_fem = float(raw["Q"])
            if raw["converged"].strip().lower() != "true":
                raise ValueError(
                    f"unconverged table node at theta={theta_deg}, log10(M)={log10_m}"
                )
            if not math.isclose(math.log10(m), log10_m, rel_tol=0.0, abs_tol=2.0e-14):
                raise ValueError("viscosity_ratio and log10_viscosity_ratio disagree")
            expected_log_c = 1.0 + math.log(math.sin(math.radians(theta_deg))) - q_fem
            if not math.isclose(
                float(raw["log_c"]), expected_log_c, rel_tol=0.0, abs_tol=5.0e-13
            ):
                raise ValueError("stored c reconstruction is inconsistent with Q")
            key = (theta_deg, log10_m)
            if key in seen:
                raise ValueError(f"duplicate table node {key}")
            seen.add(key)
            rows.append(
                Node(theta_deg, log10_m, q_fem, luo_gao_q(math.radians(theta_deg), m))
            )

    theta_values = sorted({node.theta_deg for node in rows})
    log_m_values = sorted({node.log10_m for node in rows})
    expected_theta = [30.0, 32.5, *np.arange(35.0, 146.0, 5.0).tolist(), 147.5, 150.0]
    expected_log_m = [-2.0, -1.875, -1.75, *np.arange(-1.5, 0.01, 0.25).tolist()]
    if theta_values != expected_theta or log_m_values != expected_log_m:
        raise ValueError(
            "the certified 27 x 10 half-plane grid has changed; update the plotted contract explicitly"
        )
    if len(rows) != len(theta_values) * len(log_m_values):
        raise ValueError("the FEM table is not a complete Cartesian node set")
    return sorted(rows, key=lambda node: (node.log10_m, node.theta_deg))


def node_at(nodes: list[Node], theta_deg: float, log10_m: float) -> Node:
    return next(
        node
        for node in nodes
        if node.theta_deg == theta_deg and node.log10_m == log10_m
    )


def assert_regression_anchors(nodes: list[Node]) -> None:
    """Make silent formula or table transcription drift a hard failure."""
    anchors = {
        (60.0, -1.0): (0.1212373427521456, 0.08695915976358098),
        (90.0, -1.0): (0.6106362281558018, 0.5226813589031319),
        (120.0, -1.0): (0.6897667210270724, 0.5255329402537431),
    }
    for location, expected in anchors.items():
        node = node_at(nodes, *location)
        actual = (node.q_fem, node.q_lg)
        if not all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=2.0e-12)
            for a, b in zip(actual, expected)
        ):
            raise AssertionError(
                f"finite-M anchor drift at {location}: expected {expected}, got {actual}"
            )

    delta = np.array([node.delta_q for node in nodes])
    ratio = np.array([node.c_ratio for node in nodes])
    expected_ranges = (
        -0.9552507795310332,
        -0.00710193879090637,
        1.0071272173650636,
        2.5993223580321287,
    )
    actual_ranges = (
        float(delta.min()),
        float(delta.max()),
        float(ratio.min()),
        float(ratio.max()),
    )
    if not np.allclose(actual_ranges, expected_ranges, rtol=0.0, atol=2.0e-12):
        raise AssertionError(
            f"finite-M range drift: expected {expected_ranges}, got {actual_ranges}"
        )


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(27.5, 152.5)
    axis.set_ylim(-2.08, 0.08)
    axis.set_xticks([30, 60, 90, 120, 150])
    axis.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0])
    axis.set_xlabel(r"$\theta_e\ ({}^\circ)$", fontsize=12, labelpad=5)
    axis.tick_params(
        which="major", direction="out", width=0.9, length=5, labelsize=10, pad=3
    )
    axis.tick_params(which="minor", direction="out", width=0.6, length=2.5)
    axis.minorticks_on()
    axis.grid(which="major", color="0.90", linewidth=0.55, zorder=0)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
    axis.set_box_aspect(0.72)


def make_figure(nodes: list[Node], output_dir: Path) -> tuple[Path, Path]:
    theta = np.array([node.theta_deg for node in nodes])
    log_m = np.array([node.log10_m for node in nodes])
    delta = np.array([node.delta_q for node in nodes])
    ratio = np.array([node.c_ratio for node in nodes])

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.45), sharex=True, sharey=True)
    figure.set_facecolor("white")

    # Square markers show only certified nodes.  Their positions retain the
    # deliberately nonuniform refinement at theta={32.5,147.5} and log10(M)=-1.875.
    delta_plot = axes[0].scatter(
        theta,
        log_m,
        c=delta,
        cmap="viridis",
        vmin=-1.0,
        vmax=0.0,
        marker="s",
        s=27,
        edgecolors="black",
        linewidths=0.22,
        zorder=3,
    )
    ratio_plot = axes[1].scatter(
        theta,
        log_m,
        c=ratio,
        cmap="magma",
        vmin=1.0,
        vmax=2.6,
        marker="s",
        s=27,
        edgecolors="black",
        linewidths=0.22,
        zorder=3,
    )

    for axis in axes:
        style_axis(axis)
    axes[0].set_ylabel(r"$\log_{10} M$", fontsize=12, labelpad=6)
    axes[0].text(-0.14, 1.04, r"$(a)$", transform=axes[0].transAxes, fontsize=12)
    axes[1].text(-0.14, 1.04, r"$(b)$", transform=axes[1].transAxes, fontsize=12)

    cbar_delta = figure.colorbar(delta_plot, ax=axes[0], pad=0.025, fraction=0.055)
    cbar_delta.set_label(r"$Q_{\rm LG}-Q_{\rm FEM}$", fontsize=11, labelpad=7)
    cbar_delta.ax.tick_params(labelsize=9, width=0.8, length=4)
    cbar_delta.outline.set_linewidth(0.8)
    cbar_ratio = figure.colorbar(ratio_plot, ax=axes[1], pad=0.025, fraction=0.055)
    cbar_ratio.set_label(
        r"$c_{\lambda,{\rm LG}}/c_{\lambda,{\rm FEM}}$", fontsize=11, labelpad=7
    )
    cbar_ratio.ax.tick_params(labelsize=9, width=0.8, length=4)
    cbar_ratio.outline.set_linewidth(0.8)

    figure.subplots_adjust(left=0.075, right=0.965, bottom=0.17, top=0.94, wspace=0.27)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.08,
        dpi=300,
        metadata={
            "Title": "Finite-M Luo-Gao matching comparison",
            "Creator": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.08,
        dpi=300,
        metadata={
            "Title": "Finite-M Luo-Gao matching comparison",
            "Author": "CoMPhy Lab",
        },
    )
    plt.close(figure)
    return pdf_path, png_path


def report(nodes: list[Node], pdf_path: Path, png_path: Path) -> None:
    delta = np.array([node.delta_q for node in nodes])
    ratio = np.array([node.c_ratio for node in nodes])
    worst = nodes[int(np.argmin(delta))]
    print("finite-M closure comparison")
    print("  certified nodes: 270 (27 theta values x 10 log10(M) values)")
    print(
        f"  Q_LG - Q_FEM: min={delta.min():.12g}, max={delta.max():.12g}, RMS={np.sqrt(np.mean(delta**2)):.12g}"
    )
    print(
        f"  c_LG / c_FEM: min={ratio.min():.12g}, max={ratio.max():.12g}, mean={ratio.mean():.12g}"
    )
    print(
        f"  largest mismatch: theta_e={worst.theta_deg:g} deg, log10(M)={worst.log10_m:g}"
    )
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path} (300 dpi)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = load_nodes(args.table.resolve())
    assert_regression_anchors(nodes)
    pdf_path, png_path = make_figure(nodes, args.output_dir.resolve())
    report(nodes, pdf_path, png_path)


if __name__ == "__main__":
    main()
