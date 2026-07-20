#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
# Fig. 4b reproduction plot

Overlays the pseudo-arclength/collocation continuation of the GLE
(`gle-only/gle-continuation`) on the bifurcation diagram of Fig. 4b of
Snoeijer & Andreotti, *Annu. Rev. Fluid Mech.* 45:269-292 (2013): meniscus
rise $z/\\ell_\\gamma$ versus capillary number for a partially wetting plate
withdrawn from a silicone-oil bath.

Reference data (`data/fig4b-digitized/`) was vector-extracted from the
paper's PDF (exact axis-tick calibration, no raster digitisation): the thick
grey multiscale-lubrication theory curve and the five experimental series of
Delon et al. (2008).

## Usage

```bash
uv run postProcess/plot-fig4b.py [branch.csv] [output.pdf]
# defaults: gle-only/output/fig4b-branch.csv -> img/fig4b-reproduction.png+pdf
```

## Author

Vatsal Sanjay (vatsal.sanjay@comphy-lab.org)
CoMPhy Lab, Department of Physics, Durham University
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

"""
## Paths
"""

REPO = Path(__file__).resolve().parent.parent
DIGI = REPO / 'data' / 'fig4b-digitized'

branch_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / 'gle-only' / 'output' / 'fig4b-branch.csv'
out_base = Path(sys.argv[2]).with_suffix('') if len(sys.argv) > 2 else \
    REPO / 'img' / 'fig4b-reproduction'
out_base.parent.mkdir(parents=True, exist_ok=True)

"""
## Data
"""

branch = np.genfromtxt(branch_csv, delimiter=',', names=True)
theory = np.genfromtxt(DIGI / 'theory_curve.csv', delimiter=',', names=True)

# experimental series of Delon et al. (2008): colour, marker, fill
SERIES = [
    ('symbols_red.csv', 'tab:red', 'o', 'none'),
    ('symbols_green.csv', 'tab:green', 'o', 'full'),
    ('symbols_yellow.csv', 'goldenrod', 's', 'none'),
    ('symbols_magenta.csv', 'm', 's', 'full'),
    ('symbols_blue.csv', 'tab:blue', '^', 'none'),
]

"""
## Figure
"""

fig, ax = plt.subplots(figsize=(10, 12))

# digitized theory curve of the review (thick grey)
ax.plot(theory['Ca']*1e3, theory['z_over_lgamma'], '-', color='0.55', lw=7,
        alpha=0.9, zorder=2, solid_capstyle='round',
        label=r'theory curve, digitized from Fig.~4b')

# experimental symbols
for fname, color, marker, fill in SERIES:
    d = np.genfromtxt(DIGI / fname, delimiter=',', names=True)
    kw = dict(marker=marker, ls='none', ms=11, zorder=3, alpha=0.9)
    if fill == 'none':
        kw.update(mfc='none', mec=color, mew=2.0)
    else:
        kw.update(mfc=color, mec=color)
    ax.plot(d['Ca']*1e3, d['z_over_lgamma'], **kw)

# this work: GLE continuation (C solver)
ax.plot(branch['Ca']*1e3, branch['Delta'], '-', color='k', lw=2.5, zorder=4,
        label=r'GLE continuation (this work)')

# fold marker
imax = int(np.argmax(branch['Ca']))
ax.plot(branch['Ca'][imax]*1e3, branch['Delta'][imax], 'o', ms=13,
        mfc='white', mec='k', mew=2.5, zorder=5)
ax.annotate(r'$\mathrm{Ca}^{*}$',
            (branch['Ca'][imax]*1e3, branch['Delta'][imax]),
            textcoords='offset points', xytext=(14, -2), fontsize=30)

# critical meniscus rise sqrt(2) (theta_app -> 0 at the fold)
ax.axhline(np.sqrt(2.0), color='0.75', lw=1.5, ls=':', zorder=1)
ax.text(0.25, np.sqrt(2.0) + 0.04, r'$z_c = \sqrt{2}\,\ell_\gamma$',
        fontsize=26, color='0.35')

ax.set_xlim(0, 11.8)
ax.set_ylim(0, 3.6)
ax.set_xlabel(r'$\mathrm{Ca} \times 10^{3}$', fontsize=40, labelpad=15)
ax.set_ylabel(r'$z/\ell_\gamma$', fontsize=40, labelpad=15)

ax.tick_params(which='both', direction='out', width=3, labelsize=30, pad=10)
ax.tick_params(which='major', length=12)
ax.tick_params(which='minor', length=6)
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.minorticks_on()

leg = ax.legend(fontsize=24, loc='upper left', frameon=False,
                handlelength=1.6)

# parameter annotation
imax_ca = branch['Ca'][imax]*1e3
ax.text(0.98, 0.02,
        r'$\theta_e = 53.46^\circ,\ \lambda/\ell_\gamma = 7.46\times10^{-6}$'
        '\n'
        rf'$\mathrm{{Ca}}^{{*}} = {imax_ca:.2f}\times10^{{-3}}$',
        transform=ax.transAxes, fontsize=24, ha='right', va='bottom',
        color='0.25')

plt.tight_layout()
for ext in ('.pdf', '.png'):
    plt.savefig(f'{out_base}{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'wrote {out_base}.pdf and .png')
