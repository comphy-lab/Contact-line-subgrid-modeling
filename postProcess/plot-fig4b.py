#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
# Fig. 4b reproduction and model-form comparison

Produces two deliberately separate figures from branches traced by
`reproduce-fig4b.sh`.

1. `fig4b-reproduction` overlays the digitised review curve and experiments
   with the legacy Chan branch and independently fold-calibrated Chan/Scott
   and direct Luo--Gao branches. Fold agreement is therefore a calibration,
   while branch shape and fold height remain comparisons.
2. `fig4b-model-comparison` holds the slip-length input fixed between Chan and
   Luo--Gao. No branch is re-fitted, so this plot exposes the full formulation
   difference rather than hiding it in an effective slip length.

The digitised data in `data/fig4b-digitized/` were vector-extracted from
Snoeijer & Andreotti, *Annu. Rev. Fluid Mech.* 45:269--292 (2013), Fig. 4b.

## Usage

```bash
uv run postProcess/plot-fig4b.py
uv run postProcess/plot-fig4b.py --branches-dir gle-ode/output --output-dir img
```

## Author

Vatsal Sanjay (vatsal.sanjay@comphy-lab.org)
CoMPhy Lab, Department of Physics, Durham University
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
if shutil.which("latex"):
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
else:
    matplotlib.rcParams["mathtext.fontset"] = "cm"


REPO = Path(__file__).resolve().parent.parent
DIGI = REPO / "data" / "fig4b-digitized"
PARAMS = REPO / "gle-ode"

BRANCH_FILES = {
    "legacy": "fig4b-legacy-branch.csv",
    "chan_calibrated": "fig4b-chan-scott-calibrated-branch.csv",
    "luo_gao_calibrated": "fig4b-luo-gao-calibrated-branch.csv",
    "chan_common": "fig4b-chan-common-slip-branch.csv",
    "luo_gao_common": "fig4b-luo-gao-common-slip-branch.csv",
}

PARAM_FILES = {
    "legacy": "fig4b.params",
    "chan_calibrated": "fig4b-chan-scott-calibrated.params",
    "luo_gao_calibrated": "fig4b-luo-gao-calibrated.params",
    "chan_common": "fig4b-chan-common-slip.params",
    "luo_gao_common": "fig4b-luo-gao-common-slip.params",
}

SERIES = [
    ("symbols_red.csv", "tab:red", "o", "none"),
    ("symbols_green.csv", "tab:green", "o", "full"),
    ("symbols_yellow.csv", "goldenrod", "s", "none"),
    ("symbols_magenta.csv", "m", "s", "full"),
    ("symbols_blue.csv", "tab:blue", "^", "none"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--branches-dir",
        type=Path,
        default=REPO / "gle-ode" / "output",
        help="directory containing the five named branch CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "img",
        help="directory for the reproduction and comparison figures",
    )
    return parser.parse_args()


def load_branch(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing branch {path}; run ./reproduce-fig4b.sh first"
        )
    branch = np.genfromtxt(path, delimiter=",", names=True)
    if branch.size < 3 or not {"Ca", "Delta"}.issubset(branch.dtype.names or ()):
        raise ValueError(f"invalid continuation branch: {path}")
    if np.any(~np.isfinite(branch["Ca"])) or np.any(~np.isfinite(branch["Delta"])):
        raise ValueError(f"non-finite Ca or Delta in continuation branch: {path}")
    if np.any(np.diff(branch["Delta"]) <= 0.0):
        raise ValueError(f"Delta must increase monotonically in {path}")
    return branch


def load_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def refined_fold(branch: np.ndarray) -> tuple[float, float]:
    """Match gle_colloc_march's local quadratic fold refinement."""
    imax = int(np.argmax(branch["Ca"]))
    if imax == 0 or imax == branch.size - 1:
        raise ValueError("continuation branch does not bracket its fold")
    delta0 = float(branch["Delta"][imax])
    x = branch["Delta"][imax - 1 : imax + 2] - delta0
    ca = branch["Ca"][imax - 1 : imax + 2]
    quad, linear, constant = np.polyfit(x, ca, 2)
    if not quad < 0.0:
        raise ValueError("local continuation maximum is not a concave fold")
    x_fold = -linear / (2.0 * quad)
    ca_fold = quad * x_fold**2 + linear * x_fold + constant
    return float(ca_fold), float(delta0 + x_fold)


def save_both(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axes(ax: plt.Axes, *, xlabel: str = r"$\mathrm{Ca}\ (\times 10^{-3})$") -> None:
    ax.set_xlabel(xlabel, fontsize=25, labelpad=9)
    ax.set_ylabel(r"$z/\ell_\gamma$", fontsize=25, labelpad=9)
    ax.tick_params(which="both", direction="out", width=2, labelsize=19, pad=6)
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.minorticks_on()


def plot_reproduction(
    branches: dict[str, np.ndarray],
    params: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    theory = np.genfromtxt(DIGI / "theory_curve.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(10, 12))

    ax.plot(
        theory["Ca"] * 1e3,
        theory["z_over_lgamma"],
        "-",
        color="0.6",
        lw=7,
        alpha=0.55,
        zorder=1,
        solid_capstyle="round",
        label=r"review theory curve (digitised)",
    )

    for fname, color, marker, fill in SERIES:
        data = np.genfromtxt(DIGI / fname, delimiter=",", names=True)
        kwargs = dict(marker=marker, ls="none", ms=9, zorder=2, alpha=0.8)
        if fill == "none":
            kwargs.update(mfc="none", mec=color, mew=1.6)
        else:
            kwargs.update(mfc=color, mec=color)
        ax.plot(data["Ca"] * 1e3, data["z_over_lgamma"], **kwargs)

    curves = [
        ("legacy", "0.15", "--", 2.0, r"legacy Chan, $c=3$ (fold calibrated)"),
        (
            "chan_calibrated",
            "#0066cc",
            "-",
            2.8,
            r"Chan + Scott--Hocking (fold calibrated)",
        ),
        (
            "luo_gao_calibrated",
            "#cc3311",
            "-.",
            2.8,
            r"direct Luo--Gao (fold calibrated)",
        ),
    ]
    folds: dict[str, tuple[float, float]] = {}
    for name, color, linestyle, width, label in curves:
        branch = branches[name]
        folds[name] = refined_fold(branch)
        ax.plot(
            branch["Ca"] * 1e3,
            branch["Delta"],
            color=color,
            ls=linestyle,
            lw=width,
            zorder=4,
            label=label,
        )
        ca_fold, delta_fold = folds[name]
        ax.plot(
            ca_fold * 1e3,
            delta_fold,
            "o",
            ms=9,
            mfc="white",
            mec=color,
            mew=2,
            zorder=5,
        )

    target = folds["legacy"][0]
    for name in ("chan_calibrated", "luo_gao_calibrated"):
        if not np.isclose(folds[name][0], target, rtol=1e-5, atol=5e-8):
            raise ValueError(
                f"{name} fold {folds[name][0]:.10e} misses calibration "
                f"target {target:.10e}"
            )

    ax.axhline(np.sqrt(2.0), color="0.75", lw=1.3, ls=":", zorder=0)
    ax.text(
        0.25,
        np.sqrt(2.0) + 0.04,
        r"$\theta_{\rm app}=0:\ z=\sqrt{2}\,\ell_\gamma$",
        fontsize=20,
        color="0.35",
    )
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 3.6)
    style_axes(ax)
    ax.legend(fontsize=16, loc="upper left", frameon=False, handlelength=2.3)

    chan_slip = float(params["chan_calibrated"]["slip"])
    lg_slip = float(params["luo_gao_calibrated"]["slip"])
    ax.text(
        0.98,
        0.02,
        r"$\theta_e=53.46^\circ,\ M=0$"
        "\n"
        rf"$\lambda_{{\rm Chan}}={chan_slip * 1e6:.5g}\times10^{{-6}}"
        r"\ell_\gamma$, "
        rf"$\lambda_{{\rm LG}}={lg_slip * 1e6:.6g}\times10^{{-6}}"
        r"\ell_\gamma$"
        "\n"
        r"fold position fitted; branch shape and $\Delta^*$ are checks",
        transform=ax.transAxes,
        fontsize=15,
        ha="right",
        va="bottom",
        color="0.2",
    )
    fig.tight_layout()
    save_both(fig, output_dir / "fig4b-reproduction")

    print("fold-calibrated reproduction:")
    for name in ("legacy", "chan_calibrated", "luo_gao_calibrated"):
        ca_fold, delta_fold = folds[name]
        slip = float(params[name]["slip"])
        print(
            f"  {name:18s} slip={slip:.9e} "
            f"fold_Ca={ca_fold:.10e} fold_Delta={delta_fold:.10e}"
        )


def plot_common_slip_comparison(
    branches: dict[str, np.ndarray],
    params: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    chan = branches["chan_common"]
    luo_gao = branches["luo_gao_common"]
    chan_slip = float(params["chan_common"]["slip"])
    lg_slip = float(params["luo_gao_common"]["slip"])
    if not np.isclose(chan_slip, lg_slip, rtol=0.0, atol=0.0):
        raise ValueError("fixed-input model comparison must use identical slip")

    chan_fold = refined_fold(chan)
    lg_fold = refined_fold(luo_gao)
    delta_lo = max(float(chan["Delta"].min()), float(luo_gao["Delta"].min()))
    delta_hi = min(float(chan["Delta"].max()), float(luo_gao["Delta"].max()))
    common_delta = np.linspace(delta_lo, delta_hi, 800)
    chan_ca = np.interp(common_delta, chan["Delta"], chan["Ca"])
    lg_ca = np.interp(common_delta, luo_gao["Delta"], luo_gao["Ca"])
    resolved = chan_ca >= 1e-4
    relative = 100.0 * (lg_ca[resolved] - chan_ca[resolved]) / chan_ca[resolved]

    fig, (ax, diff_ax) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax.plot(
        chan["Ca"] * 1e3,
        chan["Delta"],
        color="#0066cc",
        lw=3,
        label=r"Chan + Scott--Hocking",
    )
    ax.plot(
        luo_gao["Ca"] * 1e3,
        luo_gao["Delta"],
        color="#cc3311",
        lw=3,
        ls="-.",
        label=r"direct Luo--Gao",
    )
    for (ca_fold, delta_fold), color in (
        (chan_fold, "#0066cc"),
        (lg_fold, "#cc3311"),
    ):
        ax.plot(
            ca_fold * 1e3,
            delta_fold,
            "o",
            ms=10,
            mfc="white",
            mec=color,
            mew=2,
        )
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0.55, 3.6)
    style_axes(ax)
    ax.legend(fontsize=17, loc="upper left", frameon=False)
    ax.text(
        0.97,
        0.03,
        r"same input: $\theta_e=53.46^\circ,\ M=0$"
        "\n"
        rf"$\lambda={chan_slip * 1e6:.3g}\times10^{{-6}}\ell_\gamma$; "
        r"neither curve re-fitted",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        color="0.2",
    )

    diff_ax.axvline(0.0, color="0.65", lw=1.4, ls=":")
    diff_ax.plot(relative, common_delta[resolved], color="0.1", lw=2.5)
    style_axes(
        diff_ax,
        xlabel=r"$100(\mathrm{Ca}_{\rm LG}-\mathrm{Ca}_{\rm Chan})/"
        r"\mathrm{Ca}_{\rm Chan}\ (\%)$",
    )
    diff_ax.set_ylabel("")
    diff_ax.set_title(r"difference at equal $z/\ell_\gamma$", fontsize=18, pad=10)
    fig.tight_layout()
    save_both(fig, output_dir / "fig4b-model-comparison")

    fold_shift = 100.0 * (lg_fold[0] - chan_fold[0]) / chan_fold[0]
    print("fixed-slip model comparison:")
    print(f"  common slip       = {chan_slip:.9e}")
    print(
        f"  chan fold         = {chan_fold[0]:.10e}, "
        f"Delta={chan_fold[1]:.10e}"
    )
    print(
        f"  luo_gao fold      = {lg_fold[0]:.10e}, "
        f"Delta={lg_fold[1]:.10e}"
    )
    print(f"  relative Ca shift = {fold_shift:.6f}%")


def main() -> None:
    args = parse_args()
    branches = {
        name: load_branch(args.branches_dir / filename)
        for name, filename in BRANCH_FILES.items()
    }
    params = {
        name: load_params(PARAMS / filename) for name, filename in PARAM_FILES.items()
    }
    plot_reproduction(branches, params, args.output_dir)
    plot_common_slip_comparison(branches, params, args.output_dir)
    print(f"wrote {args.output_dir / 'fig4b-reproduction'}.pdf and .png")
    print(f"wrote {args.output_dir / 'fig4b-model-comparison'}.pdf and .png")


if __name__ == "__main__":
    main()
