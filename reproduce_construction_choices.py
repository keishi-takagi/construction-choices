#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_construction_choices.py
=================================
Replication code for

    "Construction Choices in a Reference-Based Topological Regime Classifier:
     Normalisation, Reference Selection, and What the Wasserstein Distance Measures"
    Keishi Takagi, 2026

Reproduces every table and figure in the paper -- including the path-efficiency
characterisation of S_t and the drawdown stratification behind the market-exposure
discussion -- and verifies that the published implementation reproduces
Takagi (2026a, SSRN 7057698) Table 1 cell for cell.

MIT License.

Usage
-----
    python3 reproduce_construction_choices.py \
        --hyg data/longterm_hyg.csv \
        --jnk data/longterm_jnk.csv \
        --sp500 data/longterm_sp500.csv \
        --outdir out/

Options
-------
    --normalisation {published,counting,both}   default: both
        published : uniform probability measures on the augmented diagrams,
                    eq. (2) of the paper -- the construction used in Takagi (2026a)
        counting  : counting measure, eq. (1) of the paper -- the convention in the
                    stability literature
    --surrogates N     circular-shift surrogate draws (default 2000)
    --seed N           RNG seed (default 20260819)
    --no-figures       skip figure generation
    --cache DIR        directory for persistence-diagram / distance caches

Parameters (fixed, following Takagi 2026a)
------------------------------------------
    Takens window W = 60, dimension D = 3, delay tau = 3
    Vietoris-Rips filtration threshold = 70th percentile of pairwise distances
    Rolling Z-score window = 252 trading days
    Classifier threshold z_t <= -1.5
    Forward return horizon = 20 trading days
"""

import argparse, csv, hashlib, os, pickle, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import binomtest, spearmanr, pearsonr

try:
    from ripser import ripser as _ripser
except ImportError:
    sys.exit("ripser not installed:  pip install ripser")
try:
    import ot
except ImportError:
    sys.exit("POT not installed:  pip install pot")

# ─────────────────────────────────────────────────────────────────────────
# Fixed parameters
# ─────────────────────────────────────────────────────────────────────────
W, DIM, TAU = 60, 3, 3
ZWIN, THR, HOR = 252, -1.5, 20
VR_PCTL = 70
NADIRS = {'GFC': '2009-03-09', 'European': '2011-10-03', 'COVID': '2020-03-23'}
IS_END, OOS_START = '2019-12-31', '2020-01-01'

# Muted palette
C = {'pub': '#4C6A92', 'cnt': '#A8574E', 'free': '#6E8B6B',
     'alt': '#B08D4F', 'grey': '#7F7F7F', 'light': '#C8CEDA',
     'ref': '#9A9A9A', 'dark': '#3A3A3A'}


# ─────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────
def load_series(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [r['trade_date'] for r in rows], [float(r['close']) for r in rows]


def load_map(path):
    with open(path, newline='') as f:
        return {r['trade_date']: float(r['close']) for r in csv.DictReader(f)}


# ─────────────────────────────────────────────────────────────────────────
# Topology
# ─────────────────────────────────────────────────────────────────────────
def takens_cloud(window, d=DIM, tau=TAU):
    n = len(window) - (d - 1) * tau
    if n < 3:
        return None
    pts = np.empty((n, d))
    for k in range(n):
        for j in range(d):
            pts[k, j] = window[k + j * tau]
    return (pts - pts.mean(axis=0)) / (pts.std() + 1e-10)


def h1_diagram(pts):
    """H1 persistence diagram. Returns an empty (0,2) array when no loop exists."""
    if pts is None:
        return np.zeros((0, 2))
    dm = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    thr = float(np.percentile(dm, VR_PCTL))
    try:
        dg = _ripser(pts, maxdim=1, thresh=thr)['dgms'][1]
    except Exception:
        return np.zeros((0, 2))
    if len(dg) == 0:
        return np.zeros((0, 2))
    dg = dg[np.isfinite(dg[:, 1])]
    return dg if len(dg) else np.zeros((0, 2))


def diagram_stats(dg):
    """Loop count, persistent entropy (Rucco et al. 2016), max and summed persistence."""
    if len(dg) == 0:
        return dict(loops=0, entropy=0.0, max_pers=0.0, tot_pers=0.0, sq_pers=0.0)
    p = dg[:, 1] - dg[:, 0]
    p = p[p > 0]
    if len(p) == 0:
        return dict(loops=len(dg), entropy=0.0, max_pers=0.0, tot_pers=0.0, sq_pers=0.0)
    q = p / p.sum()
    return dict(loops=len(dg), entropy=float(-np.sum(q * np.log(q + 1e-12))),
                max_pers=float(p.max()), tot_pers=float(p.sum()),
                sq_pers=float(np.sum(p ** 2)))


def wasserstein2(d1, d2, counting=True):
    """
    Wasserstein-2 distance between persistence diagrams via diagonal augmentation.

    counting=True  -> eq. (1): counting measure (stability-literature convention)
    counting=False -> eq. (2): uniform probability measures (Takagi 2026a implementation)

    The two differ by the factor n^{-1/2}, n = |D1| + |D2|.
    """
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        p = d2[:, 1] - d2[:, 0]
        c = float(np.sum(p ** 2) / 2.0)
        return np.sqrt(c) if counting else np.sqrt(c / max(len(d2), 1))
    if len(d2) == 0:
        p = d1[:, 1] - d1[:, 0]
        c = float(np.sum(p ** 2) / 2.0)
        return np.sqrt(c) if counting else np.sqrt(c / max(len(d1), 1))
    pr1 = np.column_stack([(d2[:, 0] + d2[:, 1]) / 2] * 2)
    pr2 = np.column_stack([(d1[:, 0] + d1[:, 1]) / 2] * 2)
    a = np.vstack([d1, pr1])
    b = np.vstack([d2, pr2])
    n = len(a)
    M = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
    T = ot.emd(np.ones(n) / n, np.ones(n) / n, M)
    cost = float(np.sum(T * M))          # = (counting cost) / n
    return np.sqrt(cost * n) if counting else np.sqrt(cost)


def rolling_z(x, win=ZWIN, return_sigma=False):
    z = np.full(len(x), np.nan)
    s = np.full(len(x), np.nan)
    for i in range(win, len(x)):
        w = x[i - win:i]
        mu, sd = np.nanmean(w), np.nanstd(w)
        s[i] = sd
        if sd > 1e-12:
            z[i] = (x[i] - mu) / sd
    return (z, s) if return_sigma else z


def forward_returns(dates, price_map, hor=HOR):
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        if i + hor < len(dates):
            p0 = price_map.get(d)
            p1 = price_map.get(dates[i + hor])
            if p0 and p1:
                out[i] = (p1 / p0 - 1.0) * 100.0
    return out


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────
def efficiency_ratio(window):
    """Net displacement over the window divided by the total distance travelled.

    1.0 for a monotone path, near 0 for a path that goes out and comes back.
    Computed on the same 60-day window the Takens cloud is built from, and
    scale-free, so it is comparable across the sample.
    """
    w = np.asarray(window, dtype=float)
    denom = float(np.abs(np.diff(w)).sum())
    return abs(w[-1] - w[0]) / denom if denom > 0 else np.nan


def direction_changes(window):
    """Sign reversals in the daily increments within the window."""
    s = np.sign(np.diff(np.asarray(window, dtype=float)))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if len(s) >= 2 else 0


def drawdown_from_peak(prices, win=ZWIN):
    """Drawdown from the trailing `win`-day running maximum, in per cent."""
    p = np.asarray(prices, dtype=float)
    return np.array([(p[i] / p[max(0, i - win + 1):i + 1].max() - 1.0) * 100.0
                     for i in range(len(p))])


def realised_vol(prices, idx, win):
    """Annualised realised volatility of daily log returns over the window
    ending on the previous trading day, matching the Takens window convention."""
    lr = np.diff(np.log(np.asarray(prices, dtype=float)))
    return np.array([float(np.std(lr[i - win:i - 1]) * np.sqrt(252) * 100) for i in idx])


def acf(x, maxlag=40):
    return [float(np.corrcoef(x[:-l], x[l:])[0, 1]) for l in range(1, maxlag + 1)]


def variance_inflation(a, hor=HOR):
    return float(1 + 2 * sum((1 - (l + 1) / hor) * a[l] for l in range(hor - 1)))


def activation_episodes(sel, gap=HOR):
    """Group classified days into episodes separated by more than `gap` days.

    A count of classified days overstates the independent information in a
    clustered activation pattern. The episode count is a direct, assumption-free
    complement to the variance-inflation-based effective N.
    """
    pos = np.where(sel)[0]
    if len(pos) == 0:
        return []
    eps = [[pos[0]]]
    for p in pos[1:]:
        if p - eps[-1][-1] <= gap:
            eps[-1].append(p)
        else:
            eps.append([p])
    return eps


def mahalanobis_turbulence(r, mu, S):
    """Turbulence index of Kritzman and Li (2010): the Mahalanobis distance of the
    current cross-section of returns from a reference mean and covariance.

    Expanding the quadratic form,

        D_t^2 = r_t' S^-1 r_t  -  2 mu' S^-1 r_t  +  mu' S^-1 mu,

    the third term does not depend on t and is removed exactly by a rolling
    Z-score, in the same way eq. (3) says the reference diagram is. The function
    returns the full distance, the quadratic term alone (reference mean
    discarded), and the plain squared norm (mean and covariance both discarded).
    """
    Si = np.linalg.inv(S)
    x = r - mu
    return (np.einsum('ij,jk,ik->i', x, Si, x),
            np.einsum('ij,jk,ik->i', r, Si, r),
            np.einsum('ij,ij->i', r, r),
            float(mu @ Si @ mu))


def conditional_cell(returns, base_freq, base_mean, vif):
    n = len(returns)
    if n < 5:
        return None
    k = int(np.sum(returns > 0))
    f = k / n
    return dict(n=n, eff_n=n / vif, k=k, freq=f, lift=(f - base_freq) * 100,
                mean=float(np.mean(returns)), mean_excess=float(np.mean(returns)) - base_mean,
                p=float(binomtest(k, n, base_freq).pvalue))


def evaluate(z, fwd, masks, base, vif):
    out = {}
    for label, m in masks.items():
        sel = (z <= THR) & m
        out[label] = conditional_cell(fwd[sel], base[label][0], base[label][1], vif)
    return out


def shift_null(z, mask, fwd, statfn, n_draws, rng):
    idx = np.where(~np.isnan(z))[0]
    core = z[idx]
    L = len(core)
    vals = []
    for _ in range(n_draws):
        zs = np.full(len(z), np.nan)
        zs[idx] = np.roll(core, rng.integers(1, L))
        sel = (zs <= THR) & mask
        if sel.sum() >= 5:
            vals.append(statfn(fwd[sel]))
    return np.asarray(vals)


# ─────────────────────────────────────────────────────────────────────────
# Table emission
# ─────────────────────────────────────────────────────────────────────────
class Emitter:
    """Prints to stdout, mirrors to a text file, and writes CSVs."""

    def __init__(self, outdir):
        self.tdir = os.path.join(outdir, 'tables')
        os.makedirs(self.tdir, exist_ok=True)
        self.buf = []

    def line(self, s=''):
        print(s)
        self.buf.append(s)

    def head(self, t):
        self.line()
        self.line('=' * 92)
        self.line(f'  {t}')
        self.line('=' * 92)

    def csv(self, name, header, rows):
        with open(os.path.join(self.tdir, name), 'w', newline='') as f:
            wcsv = csv.writer(f)
            wcsv.writerow(header)
            wcsv.writerows(rows)

    def save(self):
        with open(os.path.join(self.tdir, 'all_tables.txt'), 'w') as f:
            f.write('\n'.join(self.buf) + '\n')


def fmt_cell(c):
    return (f"{c['n']:>5} {c['eff_n']:>6.1f} {c['freq']*100:>7.1f}% "
            f"{c['lift']:>+7.1f} {c['p']:>7.4f}") if c else f"{'--':>5} {'--':>6} {'--':>8} {'--':>7} {'--':>7}"


# ─────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────
def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 140, 'savefig.dpi': 220, 'savefig.bbox': 'tight',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#666666', 'axes.linewidth': 0.8,
        'grid.color': '#DDDDDD', 'grid.linewidth': 0.6,
        'legend.frameon': False,
    })
    return plt


def save_fig(fig, outdir, name, plt):
    d = os.path.join(outdir, 'figures')
    os.makedirs(d, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(d, f'{name}.{ext}'))
    plt.close(fig)
    print(f'    figures/{name}.pdf')


def figure_distance_content(plt, outdir, dist, sq_pers, ncard, s_ref):
    """Fig 1: squared distance against the current diagram's own squared persistence.

    The overlaid line is the first-order prediction of eq. (5) of the paper,
    w^2 = S_t/2 + S_ref/2, with no fitted parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.2))
    for r, counting in enumerate([False, True]):
        for c, ref in enumerate(['GFC', 'European', 'COVID']):
            ax = axes[r, c]
            y = dist[(ref, counting)] ** 2
            x = sq_pers if counting else sq_pers / ncard[ref]
            col = C['cnt'] if counting else C['pub']
            ax.scatter(x, y, s=2.5, alpha=0.20, color=col, edgecolors='none', rasterized=True)
            if counting:
                xs = np.linspace(0, float(np.nanmax(x)) * 1.02, 50)
                ax.plot(xs, 0.5 * xs + 0.5 * s_ref[ref], color=C['dark'],
                        linewidth=1.0, linestyle='--',
                        label=r'$\frac{1}{2}S_t+\frac{1}{2}S_{\mathrm{ref}}$' if c == 0 else None)
                if c == 0:
                    ax.legend(loc='lower right', fontsize=7.5)
            rr = pearsonr(x, y)[0]
            ax.text(0.04, 0.94, f'$r$ = {rr:+.3f}', transform=ax.transAxes,
                    fontsize=9, va='top', color=C['dark'])
            if r == 0:
                ax.set_title(f'{ref} reference', color=C['dark'])
            if c == 0:
                lab = 'counting measure' if counting else 'uniform probability'
                ax.set_ylabel(f'{lab}\n' + r'squared distance $w_t^2$')
            ax.set_xlabel(r'$S_t=\sum_i(d_i-b_i)^2$' + ('' if counting else r'$\,/\,n_t$'),
                          fontsize=8)
            ax.grid(alpha=0.5, linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, outdir, 'fig1_distance_content', plt)


def figure_structure(plt, outdir, stats_full, stats_pub, stats_cnt):
    """Fig 2: structural statistics of classified days against the full sample."""
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    keys = [('loops', '$H_1$ loop count'),
            ('entropy', 'persistent entropy'),
            ('max_pers', 'maximum persistence')]
    labels = ['all days', 'classified\n(uniform prob.)', 'classified\n(counting)']
    cols = [C['grey'], C['pub'], C['cnt']]
    for ax, (k, title) in zip(axes, keys):
        vals = [stats_full[k], stats_pub[k], stats_cnt[k]]
        ax.bar(range(3), vals, color=cols, width=0.62)
        ax.axhline(stats_full[k], color=C['ref'], linestyle='--', linewidth=0.9, zorder=0)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.3f}'.rstrip('0').rstrip('.'), ha='center',
                    va='bottom', fontsize=8, color=C['dark'])
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, color=C['dark'])
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis='y', alpha=0.5, linewidth=0.5)
    axes[0].set_ylabel('median over classified days')
    fig.tight_layout()
    save_fig(fig, outdir, 'fig2_structure', plt)


def figure_reference_sensitivity(plt, outdir, rows):
    """Fig 3: in-sample and out-of-sample lift by configuration."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    labels = [r[0] for r in rows]
    is_l = [r[1] for r in rows]
    oos_l = [r[2] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y + 0.19, is_l, height=0.36, color=C['light'], label='in-sample ($\\leq$2019)')
    ax.barh(y - 0.19, oos_l, height=0.36, color=C['pub'], label='out-of-sample (2020--)')
    ax.axvline(0, color=C['ref'], linewidth=1.0)
    for i, (a, b) in enumerate(zip(is_l, oos_l)):
        ax.text(a + (0.7 if a >= 0 else -0.7), i + 0.19, f'{a:+.1f}', va='center',
                ha='left' if a >= 0 else 'right', fontsize=7.5, color=C['dark'])
        ax.text(b + 0.7, i - 0.19, f'{b:+.1f}', va='center', ha='left',
                fontsize=7.5, color=C['dark'])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('lift in the frequency of positive outcomes over the measured\n'
                  'unconditional reference for the period (percentage points)')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.5, linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, outdir, 'fig3_reference_sensitivity', plt)


def figure_free_and_overlap(plt, outdir, free_rows, jac_labels, jac_free, jac_mat):
    """Fig 4: reference-free classifiers and day-set overlap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8),
                                   gridspec_kw={'width_ratios': [1.0, 1.25]})
    labels = [r[0] for r in free_rows]
    oos = [r[2] for r in free_rows]
    ax1.barh(np.arange(len(labels)), oos, color=C['free'], height=0.55)
    ax1.axvline(0, color=C['ref'], linewidth=1.0)
    for i, v in enumerate(oos):
        ax1.text(v + 0.6, i, f'{v:+.1f}', va='center', fontsize=8, color=C['dark'])
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('out-of-sample lift (pp)')
    ax1.set_title('classifiers using no reference diagram', color=C['dark'])
    ax1.grid(axis='x', alpha=0.5, linewidth=0.5)

    im = ax2.imshow(jac_mat, cmap='BuPu', vmin=0, vmax=max(0.55, jac_mat.max()), aspect='auto')
    short = {'maximum persistence': 'max.\npersistence',
             'total persistence': 'total\npersistence',
             'H1 loop count': '$H_1$ loop\ncount'}
    ax2.set_xticks(range(len(jac_free)))
    ax2.set_xticklabels([short.get(c, c) for c in jac_free], fontsize=8)
    ax2.set_yticks(range(len(jac_labels)))
    ax2.set_yticklabels(jac_labels, fontsize=8)
    for i in range(jac_mat.shape[0]):
        for j in range(jac_mat.shape[1]):
            v = jac_mat[i, j]
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5,
                     color='white' if v > 0.30 else C['dark'])
    ax2.set_title('Jaccard index of classified-day sets', color=C['dark'])
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.03)
    fig.tight_layout()
    save_fig(fig, outdir, 'fig4_reference_free', plt)


def figure_shift_null(plt, outdir, panels, base_rate):
    """Fig 5: circular-shift null distributions with the observed statistic marked."""
    n = len(panels)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(10.2, 6.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, null, obs, is_freq) in zip(axes, panels):
        sc = 100 if is_freq else 1
        ax.hist(null * sc, bins=32, color=C['light'], edgecolor='white', linewidth=0.4)
        ax.axvline(obs * sc, color=C['cnt'], linewidth=1.6, label='observed')
        ax.axvline(float(np.median(null)) * sc, color=C['grey'], linestyle='--',
                   linewidth=1.0, label='null median')
        if is_freq:
            ax.axvline(base_rate * 100, color=C['free'], linestyle=':', linewidth=1.2,
                       label='measured unconditional')
        p = float(np.mean(null >= obs))
        ax.text(0.03, 0.94, f'calibrated $p$ = {p:.3f}', transform=ax.transAxes,
                fontsize=8, va='top', color=C['dark'])
        ax.set_title(title, fontsize=8.5, color=C['dark'])
        ax.set_xlabel('frequency of positive outcomes (%)' if is_freq
                      else 'mean 20-day forward return (%)')
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.4, linewidth=0.5)
    for ax in axes[len(panels):]:
        ax.axis('off')
    axes[0].legend(loc='upper left', fontsize=7.5, bbox_to_anchor=(0.02, 0.82))
    fig.tight_layout()
    save_fig(fig, outdir, 'fig5_shift_null', plt)


def figure_minpos(plt, outdir, centres, med_loops, mean_ent, ref_pts):
    """Fig 6: min_pos decile profile against diagram complexity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax1.plot(centres, med_loops, 'o-', color=C['pub'], markersize=4.5, linewidth=1.4)
    ax1.set_xlabel(r'$\mathtt{min\_pos}$ decile midpoint')
    ax1.set_ylabel('median $H_1$ loop count')
    ax1.grid(alpha=0.5, linewidth=0.5)
    ax2.plot(centres, mean_ent, 'o-', color=C['cnt'], markersize=4.5, linewidth=1.4)
    ax2.set_xlabel(r'$\mathtt{min\_pos}$ decile midpoint')
    ax2.set_ylabel('mean persistent entropy')
    ax2.grid(alpha=0.5, linewidth=0.5)
    for name, mp, lo, en in ref_pts:
        ax1.scatter([mp], [lo], marker='D', s=34, color=C['alt'],
                    edgecolors=C['dark'], linewidths=0.5, zorder=5)
        ax1.annotate(name, (mp, lo), textcoords='offset points', xytext=(4, 4),
                     fontsize=6.8, color=C['dark'])
        ax2.scatter([mp], [en], marker='D', s=34, color=C['alt'],
                    edgecolors=C['dark'], linewidths=0.5, zorder=5)
        ax2.annotate(name, (mp, en), textcoords='offset points', xytext=(4, 4),
                     fontsize=6.8, color=C['dark'])
    fig.tight_layout()
    save_fig(fig, outdir, 'fig6_minpos', plt)


def figure_efficiency(plt, outdir, S, effr, dec_S, dec_E, dec_D, cls_sets):
    """Fig 7: what a low S_t describes -- path efficiency of the 60-day window."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.4, 3.4))

    ok = ~np.isnan(effr)
    ax1.scatter(S[ok], effr[ok], s=2.5, alpha=0.16, color=C['grey'],
                edgecolors='none', rasterized=True)
    ax1.plot(dec_S, dec_E, 'o-', color=C['pub'], markersize=4.5, linewidth=1.5,
             label='decile mean')
    ax1.set_xlabel('$S_t$ (sum of squared persistences)')
    ax1.set_ylabel('path efficiency $E_t$')
    ax1.set_title('$E_t$ falls monotonically in $S_t$')
    ax1.grid(alpha=0.5, linewidth=0.5)
    ax1.legend(loc='upper right')

    x = np.arange(1, len(dec_E) + 1)
    ax2.plot(x, dec_E, 'o-', color=C['pub'], markersize=4.5, linewidth=1.5,
             label='mean $E_t$')
    ax2.set_xlabel('decile of $S_t$')
    ax2.set_ylabel('mean $E_t$', color=C['pub'])
    ax2.set_ylim(0, max(dec_E) * 1.25)
    ax2.grid(alpha=0.5, linewidth=0.5)
    axb = ax2.twinx()
    axb.plot(x, dec_D, 's--', color=C['cnt'], markersize=4, linewidth=1.2,
             label='mean direction changes')
    axb.set_ylabel('mean direction changes', color=C['cnt'])
    axb.set_ylim(0, max(dec_D) * 1.35)
    axb.spines['top'].set_visible(False)
    ax2.set_title('the reversal count is flat')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right')

    labels = [k for k, _ in cls_sets]
    means = [float(np.mean(v)) for _, v in cls_sets]
    cols = [C['grey'], C['pub'], C['cnt']][:len(means)]
    ax3.bar(range(len(means)), means, color=cols, width=0.6)
    ax3.axhline(means[0], color=C['dark'], linestyle=':', linewidth=1.0)
    ax3.set_xticks(range(len(means)))
    ax3.set_xticklabels(labels, fontsize=7.5)
    ax3.set_ylabel('mean $E_t$')
    ax3.set_title('the two constructions straddle the sample')
    ax3.grid(alpha=0.5, linewidth=0.5, axis='y')
    for i, m in enumerate(means):
        ax3.annotate(f'{m:.3f}', (i, m), textcoords='offset points',
                     xytext=(0, 3), ha='center', fontsize=7.5)

    fig.tight_layout()
    save_fig(fig, outdir, 'fig7_path_efficiency', plt)


def figure_drawdown(plt, outdir, bucket_labels, panel_pct, cls_pct, nname_list):
    """Fig 8: where the OOS classified days sit in the drawdown distribution."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(bucket_labels))
    w = 0.8 / (1 + len(cls_pct))
    ax.bar(x - 0.4 + w / 2, panel_pct, width=w, color=C['grey'],
           label='all OOS panel days')
    for j, (lab, vals) in enumerate(zip(nname_list, cls_pct)):
        ax.bar(x - 0.4 + w * (j + 1.5), vals, width=w,
               color=[C['pub'], C['cnt']][j % 2],
               label=f'classified / {lab}')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel('S\\&P 500 drawdown from the trailing 252-day peak'
                  .replace('\\&', '&'))
    ax.set_ylabel('% of the day set')
    ax.set_title('Classified days are under-represented in deep drawdowns')
    ax.grid(alpha=0.5, linewidth=0.5, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, outdir, 'fig8_drawdown_strata', plt)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hyg', default='data/longterm_hyg.csv')
    ap.add_argument('--jnk', default='data/longterm_jnk.csv')
    ap.add_argument('--sp500', default='data/longterm_sp500.csv')
    ap.add_argument('--outdir', default='out')
    ap.add_argument('--end-date', default='2026-05-15',
                    help="truncate all series at this date (YYYY-MM-DD). "
                         "The default is the data endpoint of Takagi (2026a); "
                         "pass 'none' to use every observation in the input files.")
    ap.add_argument('--normalisation', choices=['published', 'counting', 'both'],
                    default='both')
    ap.add_argument('--surrogates', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260819)
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--cache', default='.cache')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    em = Emitter(args.outdir)

    norms = ([False, True] if args.normalisation == 'both'
             else [args.normalisation == 'counting'])
    nname = {False: 'uniform prob.', True: 'counting'}

    em.line('=' * 92)
    em.line('  Construction Choices in a Reference-Based Topological Regime Classifier')
    em.line('  Replication output')
    em.line(f'  W={W}  D={DIM}  tau={TAU}  VR percentile={VR_PCTL}  Z window={ZWIN}  '
            f'threshold={THR}  horizon={HOR}')
    em.line(f'  surrogate draws={args.surrogates}  seed={args.seed}')
    em.line('=' * 92)

    # ── data ────────────────────────────────────────────────────────────
    dh, ph = load_series(args.hyg)
    dj, pj = load_series(args.jnk)
    spx = load_map(args.sp500)

    end = None if str(args.end_date).lower() in ('none', '', 'all') else args.end_date
    if end:
        def cut(dates, prices):
            keep = [i for i, d in enumerate(dates) if d <= end]
            return [dates[i] for i in keep], [prices[i] for i in keep]
        dh, ph = cut(dh, ph)
        dj, pj = cut(dj, pj)
        spx = {d: v for d, v in spx.items() if d <= end}
    em.line(f'\n  Sample frame: {"through " + end if end else "all available observations"}')
    em.line(f'    HYG {dh[0]} to {dh[-1]}  ({len(dh)} rows)')
    em.line(f'    JNK {dj[0]} to {dj[-1]}  ({len(dj)} rows)')

    # ── persistence diagrams ────────────────────────────────────────────
    # The cache key is derived from everything that determines the cache's
    # contents: the sample endpoint, the geometry parameters, and a digest of the
    # price series actually loaded. Keying on the endpoint alone lets a cache
    # built from one data vintage be reused silently against another, which mixes
    # two vintages inside a single run.
    dtag = hashlib.sha256(
        repr((dh, ph, dj, pj, sorted(spx.items()), W, DIM, TAU, VR_PCTL)).encode()
    ).hexdigest()[:12]
    ftag = f"{(end or 'all').replace('-', '')}_{dtag}"
    em.line(f'  input digest: {dtag}')
    pdcache = os.path.join(args.cache, f'pd_{ftag}.pkl')
    if os.path.exists(pdcache):
        with open(pdcache, 'rb') as f:
            PD = pickle.load(f)
        em.line('\n  persistence-diagram cache loaded')
    else:
        PD = {}
        for tag, dates, prices in [('HYG', dh, ph), ('JNK', dj, pj)]:
            em.line(f'\n  computing {tag} persistence diagrams ({len(dates)-W}) ...')
            store = {}
            for i in range(W, len(dates)):
                dg = h1_diagram(takens_cloud(prices[i - W:i]))
                st = diagram_stats(dg)
                win = np.asarray(prices[i - W:i])
                st['min_pos'] = int(np.argmin(win)) / (len(win) - 1)
                st['eff_ratio'] = efficiency_ratio(win)
                st['dir_chg'] = direction_changes(win)
                st['pd'] = dg
                store[i] = st
            PD[tag] = store
        with open(pdcache, 'wb') as f:
            pickle.dump(PD, f)

    idx_h = sorted(PD['HYG'].keys())
    dates_h = [dh[i] for i in idx_h]
    da = np.asarray(dates_h)
    fwd = forward_returns(dates_h, spx)

    ok = ~np.isnan(fwd)

    # The classifier is a 252-day rolling Z-score of a series that itself begins
    # at the W-th trading day, so no construction -- reference-based or
    # reference-free -- can mark a day before index ZWIN. Measuring the base rate
    # on days the classifier cannot reach compares the conditional frequency
    # against a population it was never drawn from. The definition below is
    # deliberately construction-independent: it depends on W and ZWIN only, so
    # the same panel serves every classifier in the paper.
    reachable = np.zeros(len(idx_h), dtype=bool)
    reachable[ZWIN:] = True

    masks = {'Full': ok & reachable,
             'IS': ok & reachable & (da <= IS_END),
             'OOS': ok & reachable & (da >= OOS_START)}
    base = {k: (float(np.mean(fwd[m] > 0)), float(np.mean(fwd[m]))) for k, m in masks.items()}

    # Retained for the descriptive diagram-level statistics of Sections 9.2-9.4,
    # which involve no base rate and are reported over every computable diagram.
    masks_all = {'Full': ok, 'IS': ok & (da <= IS_END), 'OOS': ok & (da >= OOS_START)}
    base_all = {k: (float(np.mean(fwd[m] > 0)), float(np.mean(fwd[m])))
                for k, m in masks_all.items()}

    S = np.array([PD['HYG'][i]['sq_pers'] for i in idx_h])
    loops = np.array([PD['HYG'][i]['loops'] for i in idx_h], float)
    entro = np.array([PD['HYG'][i]['entropy'] for i in idx_h])
    maxp = np.array([PD['HYG'][i]['max_pers'] for i in idx_h])
    totp = np.array([PD['HYG'][i]['tot_pers'] for i in idx_h])
    minp = np.array([PD['HYG'][i]['min_pos'] for i in idx_h])
    effr = np.array([PD['HYG'][i]['eff_ratio'] for i in idx_h], dtype=float)
    dchg = np.array([PD['HYG'][i]['dir_chg'] for i in idx_h], dtype=float)

    def reference(tag, dates, nadir):
        target = datetime.strptime(nadir, '%Y-%m-%d')
        for off in range(15):
            d2 = (target + timedelta(days=off)).strftime('%Y-%m-%d')
            if d2 in dates and dates.index(d2) in PD[tag]:
                return PD[tag][dates.index(d2)]['pd'], d2
        raise RuntimeError(f'reference not found: {tag} {nadir}')

    REF = {}
    for tag, dates in [('HYG', dh), ('JNK', dj)]:
        for nm, nd in NADIRS.items():
            REF[(tag, nm)] = reference(tag, dates, nd)

    # ── distances ───────────────────────────────────────────────────────
    dcache = os.path.join(args.cache, f'dist_{VR_PCTL}_{ftag}.pkl')
    if os.path.exists(dcache):
        with open(dcache, 'rb') as f:
            DIST = pickle.load(f)
        em.line('  distance cache loaded')
    else:
        DIST = {}
        for nm in NADIRS:
            rpd = REF[('HYG', nm)][0]
            for cnt in [False, True]:
                em.line(f'  computing distances: {nm} / {nname[cnt]} ...')
                DIST[(nm, cnt)] = np.array(
                    [wasserstein2(PD['HYG'][i]['pd'], rpd, counting=cnt) for i in idx_h])
        with open(dcache, 'wb') as f:
            pickle.dump(DIST, f)

    Z = {k: rolling_z(v) for k, v in DIST.items()}
    Zfree = {'maximum persistence': rolling_z(maxp),
             'total persistence': rolling_z(totp),
             '$H_1$ loop count': rolling_z(loops)}

    # ══ Table 1: panel and base rates ═══════════════════════════════════
    em.head('Table 1  Analysis panel and measured unconditional reference')
    em.line('  Measured on the reachable panel: days on which some classifier could')
    em.line(f'  fire, i.e. after the {W}-day Takens warm-up and the {ZWIN}-day Z-score')
    em.line('  warm-up. Every construction in this paper, reference-based and')
    em.line('  reference-free alike, becomes computable on the same day.\n')
    em.line(f"  {'Period':<14}{'n':>8}{'Pos. freq.':>13}{'Mean':>11}")
    em.line('  ' + '-' * 46)
    rows = []
    for k, lab in [('Full', 'Full'), ('IS', 'IS (<=2019)'), ('OOS', 'OOS (2020-)')]:
        n = int(masks[k].sum())
        em.line(f'  {lab:<14}{n:>8}{base[k][0]*100:>12.2f}%{base[k][1]:>+10.3f}%')
        rows.append([lab, n, round(base[k][0] * 100, 2), round(base[k][1], 3)])
    em.csv('table1_base_rates.csv', ['period', 'n', 'pos_freq_pct', 'mean_pct'], rows)
    em.line(f'\n  Nominal 50% reference understates the applicable null by '
            f'{(base["Full"][0]-0.5)*100:.1f} pp.')
    em.line(f'  Nominal zero reference for the mean understates it by '
            f'{base["Full"][1]:.3f} pp.')

    excl = ok & ~reachable
    if excl.sum():
        em.line(f'\n  Excluded as unreachable: {int(excl.sum())} days, '
                f'{dates_h[int(np.argmax(excl))]} to '
                f'{dates_h[int(np.where(excl)[0][-1])]},')
        em.line(f'  whose frequency of positive outcomes is '
                f'{np.mean(fwd[excl]>0)*100:.2f}%. Measuring the reference on the '
                f'full\n  {int(ok.sum())}-day panel instead would give '
                f'{base_all["Full"][0]*100:.2f}% (Full), '
                f'{base_all["IS"][0]*100:.2f}% (IS), '
                f'{base_all["OOS"][0]*100:.2f}% (OOS),')
        em.line('  and would compare the conditional frequency against a population')
        em.line('  the classifier cannot draw from.')
        em.csv('table1b_reachability.csv',
               ['quantity', 'value'],
               [['reachable_n', int(masks['Full'].sum())],
                ['unreachable_n', int(excl.sum())],
                ['unreachable_start', dates_h[int(np.argmax(excl))]],
                ['unreachable_end', dates_h[int(np.where(excl)[0][-1])]],
                ['unreachable_pos_freq_pct', round(float(np.mean(fwd[excl] > 0) * 100), 2)],
                ['full_panel_pos_freq_pct', round(base_all['Full'][0] * 100, 2)],
                ['full_panel_is_pos_freq_pct', round(base_all['IS'][0] * 100, 2)]])

    # ══ Table 2: dependence ═════════════════════════════════════════════
    a = acf(fwd[ok])
    vif = variance_inflation(a)
    zero_cross = next((l + 1 for l, v in enumerate(a) if v <= 0), None)
    em.head('Table 2  Dependence structure of the overlapping outcome variable')
    em.line(f'  ACF of r_t,20:  lag1={a[0]:.3f}  lag5={a[4]:.3f}  lag10={a[9]:.3f}  '
            f'lag19={a[18]:.3f}  lag20={a[19]:.3f}')
    em.line(f'  zero crossing at lag {zero_cross}')
    em.line(f'  Bartlett-truncated variance-inflation factor = {vif:.2f}')
    em.line(f'  effective N = nominal / {vif:.1f}')
    em.csv('table2_dependence.csv', ['lag', 'acf'],
           [[l + 1, round(v, 4)] for l, v in enumerate(a)])

    # ══ Table 3: normalisation sensitivity + verification ═══════════════
    em.head('Table 3  Normalisation sensitivity (GFC nadir reference)')
    em.line(f"  {'Construction':<20}{'Period':<7}{'n':>5}{'effN':>7}{'Pos.freq':>9}"
            f"{'Lift':>8}{'p':>8}{'Mean':>9}{'Excess':>9}")
    em.line('  ' + '-' * 84)
    rows = []
    res = {}
    for cnt in norms:
        res[cnt] = evaluate(Z[('GFC', cnt)], fwd, masks, base, vif)
        for k in ['Full', 'IS', 'OOS']:
            c = res[cnt][k]
            em.line(f"  {nname[cnt]:<20}{k:<7}{c['n']:>5}{c['eff_n']:>7.1f}"
                    f"{c['freq']*100:>8.1f}%{c['lift']:>+8.1f}{c['p']:>8.4f}"
                    f"{c['mean']:>+8.2f}%{c['mean_excess']:>+8.2f}")
            rows.append([nname[cnt], k, c['n'], round(c['eff_n'], 1),
                         round(c['freq'] * 100, 1), round(c['lift'], 1), round(c['p'], 4),
                         round(c['mean'], 2), round(c['mean_excess'], 2)])
    em.csv('table3_normalisation.csv',
           ['construction', 'period', 'n', 'eff_n', 'pos_freq_pct', 'lift_pp', 'p',
            'mean_pct', 'mean_excess_pp'], rows)

    if False in res:
        em.line('\n  Verification against Takagi (2026a, SSRN 7057698) Table 1')
        em.line('  (the published implementation is the uniform-probability construction)')
        exp = {'Full': (84, 2.98, 77.4), 'IS': (54, 2.76, 70.4), 'OOS': (30, 3.38, 90.0)}
        allok = True
        for k, (en_, em_, ef_) in exp.items():
            c = res[False][k]
            good = (c['n'] == en_ and abs(c['mean'] - em_) < 0.015
                    and abs(c['freq'] * 100 - ef_) < 0.1)
            allok &= good
            em.line(f"    {k:<5} n {c['n']:>3} (exp {en_:>3})   "
                    f"mean {c['mean']:+.2f}% (exp {em_:+.2f}%)   "
                    f"freq {c['freq']*100:.1f}% (exp {ef_:.1f}%)   "
                    f"{'PASS' if good else 'FAIL'}")
        if allok:
            em.line('  -> PASS: reproduces Takagi (2026a) Table 1 cell for cell')
        else:
            em.line('  -> FAIL')
            em.line('     Takagi (2026a) is computed on daily data through 2026-05-15 as')
            em.line('     retrieved at that time. Two things break an exact match:')
            em.line(f'       (a) sample frame. This run ends {dh[-1]}; pass')
            em.line("           --end-date 2026-05-15 to match the source paper.")
            em.line('       (b) data vintage. The provider re-bases its dividend adjustment,')
            em.line('           which rescales the whole history by a constant. That is')
            em.line('           topologically inert here -- the Takens cloud is scale-')
            em.line('           normalised -- but it changes the fourth decimal the prices')
            em.line('           are stored in, and Table 15 below shows how many days sit')
            em.line('           close enough to the threshold to be decided by that digit.')
            em.line('     A discrepancy of one or two classified days is expected under (b)')
            em.line('     and is reported in the paper (Section 12, data vintage and')
            em.line('     threshold marginality). Compare Table 5 against the manuscript to')
            em.line('     confirm the input series rather than the code.')

    # ══ Table 4: what the distance measures ═════════════════════════════
    em.head('Table 4  Correlation of the squared distance with the current diagram alone')
    em.line('  S_t = sum of squared persistences of D_t; no reference information used.')
    em.line(f"\n  {'Reference':<12}{'Construction':<16}{'corr(w2,S)':>12}"
            f"{'corr(w2,S/n)':>14}{'corr(w,sqrt S)':>16}{'corr(z_w,z_S)':>16}")
    em.line('  ' + '-' * 88)
    rows = []
    ncard = {}
    for nm in NADIRS:
        rlen = len(REF[('HYG', nm)][0])
        ncard[nm] = loops + rlen
        for cnt in norms:
            w = DIST[(nm, cnt)]
            r1 = pearsonr(w ** 2, S)[0]
            r2 = pearsonr(w ** 2, S / ncard[nm])[0]
            r3 = pearsonr(w, np.sqrt(S))[0]
            # The correlations above are in levels; the classifier acts on the
            # Z-scored series. rz is the correlation the threshold actually sees.
            za = rolling_z(w)
            zb = rolling_z(S / ncard[nm] if not cnt else S)
            fin = ~np.isnan(za) & ~np.isnan(zb)
            rz = pearsonr(za[fin], zb[fin])[0]
            em.line(f'  {nm:<12}{nname[cnt]:<16}{r1:>+12.3f}{r2:>+14.3f}'
                    f'{r3:>+16.3f}{rz:>+16.3f}')
            rows.append([nm, nname[cnt], round(r1, 3), round(r2, 3), round(r3, 3),
                         round(rz, 3)])
    em.csv('table4_distance_content.csv',
           ['reference', 'construction', 'corr_w2_S', 'corr_w2_S_over_n', 'corr_w_sqrtS',
            'corr_z_w_z_S'], rows)
    em.line('\n  The column matching each construction is corr(w2,S/n) for the uniform-')
    em.line('  probability form and corr(w2,S) for the counting-measure form.')
    em.line('  corr(z_w, z_S) is the correlation after both series are put through the')
    em.line('  same 252-day rolling Z-score. It is the association the threshold sees,')
    em.line('  and it is the right figure to read against the Jaccard indices of')
    em.line('  Table 8: a high correlation in levels does not imply agreement in the')
    em.line('  tail where the threshold operates.')

    # ══ Table 5: reference structure ════════════════════════════════════
    em.head('Table 5  Structure of the reference diagrams')
    em.line(f"  {'ETF':<5}{'Reference':<12}{'Date':<13}{'loops':>7}{'entropy':>10}"
            f"{'max pers':>11}{'min_pos':>10}")
    em.line('  ' + '-' * 70)
    rows = []
    for tag, dates in [('HYG', dh), ('JNK', dj)]:
        for nm in NADIRS:
            dg, used = REF[(tag, nm)]
            st = diagram_stats(dg)
            mp = PD[tag][dates.index(used)]['min_pos']
            em.line(f"  {tag:<5}{nm:<12}{used:<13}{st['loops']:>7}{st['entropy']:>10.3f}"
                    f"{st['max_pers']:>11.4f}{mp:>10.3f}")
            rows.append([tag, nm, used, st['loops'], round(st['entropy'], 3),
                         round(st['max_pers'], 4), round(mp, 3)])
    em.csv('table5_reference_structure.csv',
           ['etf', 'reference', 'date', 'loops', 'entropy', 'max_pers', 'min_pos'], rows)

    # ══ Table 6: reference selection ════════════════════════════════════
    em.head('Table 6  Reference selection sensitivity (HYG price series)')
    em.line(f"  {'Configuration':<28}" +
            ''.join([f'{p:^25}' for p in ['Full', 'IS', 'OOS']]))
    em.line(f"  {'':<28}" + ''.join([f"{'n':>5}{'effN':>6}{'freq':>8}{'lift':>7}"
                                     for _ in range(3)]))
    em.line('  ' + '-' * 104)
    rows, fig3rows = [], []
    for cnt in norms:
        for nm in NADIRS:
            ev = evaluate(Z[(nm, cnt)], fwd, masks, base, vif)
            label = f'{nm} / {nname[cnt]}'
            seg = ''
            for k in ['Full', 'IS', 'OOS']:
                c = ev[k]
                seg += (f"{c['n']:>5}{c['eff_n']:>6.1f}{c['freq']*100:>7.1f}%"
                        f"{c['lift']:>+7.1f}") if c else f"{'--':>5}{'--':>6}{'--':>8}{'--':>7}"
            em.line(f'  {label:<28}{seg}')
            rows.append([nm, nname[cnt]] +
                        [x for k in ['Full', 'IS', 'OOS']
                         for x in ([ev[k]['n'], round(ev[k]['eff_n'], 1),
                                    round(ev[k]['freq'] * 100, 1), round(ev[k]['lift'], 1)]
                                   if ev[k] else ['', '', '', ''])])
            fig3rows.append((label, ev['IS']['lift'] if ev['IS'] else 0,
                             ev['OOS']['lift'] if ev['OOS'] else 0))
    em.csv('table6_reference_selection.csv',
           ['reference', 'construction'] +
           [f'{p}_{s}' for p in ['full', 'is', 'oos'] for s in ['n', 'eff_n', 'freq', 'lift']],
           rows)

    # ══ Table 7: reference-free ═════════════════════════════════════════
    em.head('Table 7  Classifiers using no reference diagram')
    em.line(f"  {'Summary statistic':<24}" +
            ''.join([f"{'n':>6}{'effN':>7}{'lift':>8}" for _ in range(3)]))
    em.line(f"  {'':<24}" + ''.join([f'{p:^21}' for p in ['Full', 'IS', 'OOS']]))
    em.line('  ' + '-' * 88)
    rows, fig4rows = [], []
    free_ev = {}
    for label, z in Zfree.items():
        ev = evaluate(z, fwd, masks, base, vif)
        free_ev[label] = ev
        seg = ''.join([(f"{ev[k]['n']:>6}{ev[k]['eff_n']:>7.1f}{ev[k]['lift']:>+8.1f}"
                        if ev[k] else f"{'--':>6}{'--':>7}{'--':>8}") for k in ['Full', 'IS', 'OOS']])
        plain = label.replace('$H_1$ ', 'H1 ')
        em.line(f'  {plain:<24}{seg}')
        rows.append([plain] + [x for k in ['Full', 'IS', 'OOS']
                               for x in ([ev[k]['n'], round(ev[k]['eff_n'], 1),
                                          round(ev[k]['lift'], 1)] if ev[k] else ['', '', ''])])
        fig4rows.append((plain, ev['Full']['lift'] if ev['Full'] else 0,
                         ev['OOS']['lift'] if ev['OOS'] else 0))
    em.csv('table7_reference_free.csv',
           ['statistic'] + [f'{p}_{s}' for p in ['full', 'is', 'oos']
                            for s in ['n', 'eff_n', 'lift']], rows)

    # ══ Table 8: Jaccard ════════════════════════════════════════════════
    em.head('Table 8  Jaccard index between classified-day sets')
    free_sets = {}
    for label, z in Zfree.items():
        free_sets[label.replace('$H_1$ ', 'H1 ')] = set(np.where((z <= THR) & ~np.isnan(z))[0])
    fcols = list(free_sets.keys())
    em.line(f"  {'Configuration':<28}" + ''.join([f'{c:>22}' for c in fcols]))
    em.line('  ' + '-' * 96)
    rows, jl, jm = [], [], []
    for cnt in norms:
        for nm in NADIRS:
            z = Z[(nm, cnt)]
            s = set(np.where((z <= THR) & ~np.isnan(z))[0])
            vals = [len(s & fs) / len(s | fs) for fs in free_sets.values()]
            label = f'{nm} / {nname[cnt]}'
            em.line(f'  {label:<28}' + ''.join([f'{v:>22.3f}' for v in vals]))
            rows.append([nm, nname[cnt]] + [round(v, 3) for v in vals])
            jl.append(label)
            jm.append(vals)
    em.csv('table8_jaccard.csv', ['reference', 'construction'] + fcols, rows)

    # ══ Table 9: shift null ═════════════════════════════════════════════
    em.head(f'Table 9  Circular-shift surrogate test ({args.surrogates} draws)')
    em.line(f"  {'Classifier':<30}{'Statistic':<12}{'Observed':>10}{'Null med':>10}"
            f"{'Null p95':>10}{'Calib. p':>10}")
    em.line('  ' + '-' * 84)
    # All six reference x normalisation configurations are tested, not a subset.
    # A claim of the form "no reference-based configuration is distinguishable"
    # is a statement about all six and has to be computed on all six.
    targets = [(f'{nm} / {nname[c]}', Z[(nm, c)]) for c in norms for nm in NADIRS]
    targets += [(f'{k} (no reference)', v) for k, v in
                [(kk.replace('$H_1$ ', 'H1 '), vv) for kk, vv in Zfree.items()]
                if k in ('maximum persistence', 'total persistence')]
    stat_defs = [('Full freq.', 'Full', lambda r: float(np.mean(r > 0)), True),
                 ('OOS freq.', 'OOS', lambda r: float(np.mean(r > 0)), True),
                 ('Full mean', 'Full', lambda r: float(np.mean(r)), False)]
    rows, fig5panels = [], []
    for cname, z in targets:
        for sname, mkey, sfn, isfreq in stat_defs:
            obs = sfn(fwd[(z <= THR) & masks[mkey]])
            null = shift_null(z, masks[mkey], fwd, sfn, args.surrogates, rng)
            p = float(np.mean(null >= obs))
            sc = 100 if isfreq else 1
            em.line(f'  {cname:<30}{sname:<12}{obs*sc:>10.1f}{np.median(null)*sc:>10.1f}'
                    f'{np.percentile(null,95)*sc:>10.1f}{p:>10.4f}')
            rows.append([cname, sname, round(obs * sc, 2), round(float(np.median(null)) * sc, 2),
                         round(float(np.percentile(null, 95)) * sc, 2), round(p, 4)])
            if cname.startswith('GFC') or (cname.startswith('total') and sname == 'OOS freq.'):
                fig5panels.append((f'{cname}\n{sname}', null, obs, isfreq))
        em.line('')
    em.csv('table9_shift_null.csv',
           ['classifier', 'statistic', 'observed', 'null_median', 'null_p95', 'calibrated_p'],
           rows)
    med_full = [r[3] for r in rows if r[1] == 'Full freq.']
    med_oos = [r[3] for r in rows if r[1] == 'OOS freq.']
    em.line(f'  Null medians, Full freq.: {min(med_full):.1f} to {max(med_full):.1f}% '
            f'against a measured {base["Full"][0]*100:.2f}%.')
    em.line(f'  Null medians, OOS freq. : {min(med_oos):.1f} to {max(med_oos):.1f}% '
            f'against a measured {base["OOS"][0]*100:.2f}%.')
    em.line('  The procedure recovers the reference without being supplied it. The')
    em.line('  OOS medians sit above it because a shifted classifier places fewer')
    em.line('  days in the OOS window, and the median of a small-n binomial')
    em.line('  proportion is upward-biased by discreteness; read the Full-freq.')
    em.line('  column for the recovery property.')

    # ══ Table 10: joint grid ════════════════════════════════════════════
    em.head('Table 10  Joint agreement of the configurations under a common circular shift')
    grid_ref = [(f'{nm} / {nname[c]}', Z[(nm, c)]) for c in norms for nm in NADIRS]
    grid_free = [('maximum persistence (no ref.)', Zfree['maximum persistence']),
                 ('total persistence (no ref.)', Zfree['total persistence'])]
    grid = grid_ref + grid_free
    obs_lift = []
    for nm_, z_ in grid:
        sel = (z_ <= THR) & masks['OOS']
        obs_lift.append((float(np.mean(fwd[sel] > 0)) - base['OOS'][0]) * 100)
    em.line(f'  {len(grid_ref)} reference-based configurations and {len(grid_free)} '
            f'reference-free classifiers are evaluated jointly.')
    for (nm_, _), v in zip(grid, obs_lift):
        em.line(f'    {nm_:<34}{v:>+7.1f} pp')
    gidx = {nm_: np.where(~np.isnan(z_))[0] for nm_, z_ in grid}
    counts = {t: [] for t in (15, 20, 25)}
    counts_ref = {t: [] for t in (15, 20, 25)}
    stat_free = {'min': [], 'median': []}
    stat_free_ref = {'min': [], 'median': []}
    n_ref = len(grid_ref)
    for _ in range(args.surrogates):
        sh = rng.integers(1, 4000)
        lifts = []
        for nm_, z_ in grid:
            ii = gidx[nm_]
            zs = np.full(len(z_), np.nan)
            zs[ii] = np.roll(z_[ii], sh)
            sel = (zs <= THR) & masks['OOS']
            lifts.append((float(np.mean(fwd[sel] > 0)) - base['OOS'][0]) * 100
                         if sel.sum() >= 5 else -99.0)
        for t in counts:
            counts[t].append(sum(1 for v in lifts if v >= t))
            counts_ref[t].append(sum(1 for v in lifts[:n_ref] if v >= t))
        stat_free['min'].append(min(lifts))
        stat_free['median'].append(float(np.median(lifts)))
        stat_free_ref['min'].append(min(lifts[:n_ref]))
        stat_free_ref['median'].append(float(np.median(lifts[:n_ref])))
    em.line(f"\n  {'Family':<26}{'Threshold':<12}{'Observed':>12}{'Null median':>14}"
            f"{'P(null>=obs)':>15}")
    em.line('  ' + '-' * 80)
    rows = []
    for fam, cd, sub in [('reference-based only', counts_ref, obs_lift[:n_ref]),
                         ('all classifiers', counts, obs_lift)]:
        for t in (15, 20, 25):
            cnts = np.asarray(cd[t])
            o = sum(1 for v in sub if v >= t)
            p = float(np.mean(cnts >= o))
            em.line(f'  {fam:<26}>= +{t} pp{"":<3}{o:>7} of {len(sub)}'
                    f'{np.median(cnts):>14.1f}{p:>15.4f}')
            rows.append([fam, t, o, len(sub), float(np.median(cnts)), round(p, 4)])
        em.line('')
    em.csv('table10_joint_grid.csv',
           ['family', 'lift_threshold_pp', 'observed_count', 'n_configs',
            'null_median', 'p'], rows)
    em.line('\n  The same shift offset is applied to every configuration in each draw,')
    em.line('  preserving the cross-configuration correlation structure.')
    em.line('  Applying independent shifts would understate the null.')

    # ── threshold-free version of the same test ─────────────────────────
    em.line('\n  A count of how many configurations clear a bar is a function of where')
    em.line('  the bar is put, and the bar is ours. Below, the same null is applied')
    em.line('  to order statistics of the lift vector, which introduce no threshold.')
    em.line(f"\n  {'Family':<26}{'Statistic':<14}{'Observed':>11}{'Null median':>14}"
            f"{'P(null>=obs)':>15}")
    em.line('  ' + '-' * 80)
    rows = []
    for fam, sf, sub in [('reference-based only', stat_free_ref, obs_lift[:n_ref]),
                         ('all classifiers', stat_free, obs_lift)]:
        for sname, ofn in [('minimum lift', min), ('median lift', np.median)]:
            key = 'min' if sname.startswith('min') else 'median'
            o = float(ofn(sub))
            nullv = np.asarray(sf[key])
            p = float(np.mean(nullv >= o))
            em.line(f'  {fam:<26}{sname:<14}{o:>+10.1f}{np.median(nullv):>+14.1f}'
                    f'{p:>15.4f}')
            rows.append([fam, sname, round(o, 2), round(float(np.median(nullv)), 2),
                         round(p, 4)])
        em.line('')
    em.csv('table10b_joint_threshold_free.csv',
           ['family', 'statistic', 'observed_pp', 'null_median_pp', 'p'], rows)
    em.line('  The minimum lift is the conservative summary: it asks whether even the')
    em.line('  weakest configuration exceeds what a dependence-preserving null')
    em.line('  produces. It carries no threshold choice.')

    # ══ Table 11: structural statistics of classified days ══════════════
    em.head('Table 11  Structural statistics of classified days (GFC reference)')
    valid = ~np.isnan(Z[('GFC', norms[0])])
    full_stats = {'loops': float(np.median(loops[valid])),
                  'entropy': float(np.median(entro[valid])),
                  'max_pers': float(np.median(maxp[valid]))}
    em.line(f"  {'Set':<28}{'loops':>9}{'entropy':>11}{'max pers':>12}")
    em.line('  ' + '-' * 62)
    em.line(f"  {'all days':<28}{full_stats['loops']:>9.2f}{full_stats['entropy']:>11.3f}"
            f"{full_stats['max_pers']:>12.4f}")
    rows = [['all days', round(full_stats['loops'], 2), round(full_stats['entropy'], 3),
             round(full_stats['max_pers'], 4)]]
    cls_stats = {}
    for cnt in norms:
        m = (Z[('GFC', cnt)] <= THR) & valid
        st = {'loops': float(np.median(loops[m])), 'entropy': float(np.median(entro[m])),
              'max_pers': float(np.median(maxp[m]))}
        cls_stats[cnt] = st
        em.line(f"  {'classified / ' + nname[cnt]:<28}{st['loops']:>9.2f}"
                f"{st['entropy']:>11.3f}{st['max_pers']:>12.4f}")
        rows.append([f'classified / {nname[cnt]}', round(st['loops'], 2),
                     round(st['entropy'], 3), round(st['max_pers'], 4)])
    em.line('\n  Direction required by the geometric reading in Takagi (2026a, Sec. 6.1):')
    em.line('    loops lower, entropy lower, maximum persistence higher.')
    for cnt in norms:
        st = cls_stats[cnt]
        j = ['yes' if st['loops'] < full_stats['loops'] else 'no',
             'yes' if st['entropy'] < full_stats['entropy'] else 'no',
             'yes' if st['max_pers'] > full_stats['max_pers'] else 'no']
        em.line(f'    {nname[cnt]:<16} loops {j[0]:<5} entropy {j[1]:<5} max pers {j[2]}')
    em.csv('table11_structure.csv', ['set', 'loops', 'entropy', 'max_pers'], rows)

    em.line('\n  Correlation of the raw distance with structural statistics:')
    em.line(f"  {'Construction':<20}{'loops':>10}{'entropy':>11}{'max pers':>12}")
    rows2 = []
    for cnt in norms:
        w = DIST[('GFC', cnt)]
        r = [pearsonr(w, loops)[0], pearsonr(w, entro)[0], pearsonr(w, maxp)[0]]
        em.line(f'  {nname[cnt]:<20}{r[0]:>+10.3f}{r[1]:>+11.3f}{r[2]:>+12.3f}')
        rows2.append([nname[cnt]] + [round(v, 3) for v in r])
    em.csv('table11b_distance_structure.csv',
           ['construction', 'corr_loops', 'corr_entropy', 'corr_max_pers'], rows2)

    # ══ Table 12: min_pos ═══════════════════════════════════════════════
    em.head('Table 12  The within-window minimum-price position')
    em.line(f'  Pearson  corr(min_pos, loops)   = {pearsonr(minp, loops)[0]:+.3f}')
    em.line(f'  Spearman corr(min_pos, loops)   = {spearmanr(minp, loops).statistic:+.3f}')
    em.line(f'  Spearman corr(min_pos, entropy) = {spearmanr(minp, entro).statistic:+.3f}')
    q = np.quantile(minp, np.linspace(0, 1, 11))
    em.line(f"\n  {'Decile':<22}{'n':>8}{'median loops':>15}{'mean entropy':>15}")
    em.line('  ' + '-' * 62)
    rows, centres, mloops, ment = [], [], [], []
    for i in range(10):
        m = (minp >= q[i]) & ((minp <= q[i + 1]) if i == 9 else (minp < q[i + 1]))
        if m.sum() == 0:
            continue
        ml, me = float(np.median(loops[m])), float(np.mean(entro[m]))
        em.line(f'  [{q[i]:.3f}, {q[i+1]:.3f}]{"":<6}{int(m.sum()):>8}{ml:>15.1f}{me:>15.3f}')
        rows.append([round(q[i], 3), round(q[i + 1], 3), int(m.sum()), ml, round(me, 3)])
        centres.append((q[i] + q[i + 1]) / 2)
        mloops.append(ml)
        ment.append(me)
    em.csv('table12_minpos_deciles.csv',
           ['decile_lo', 'decile_hi', 'n', 'median_loops', 'mean_entropy'], rows)

    em.line('\n  min_pos alone as a classifier:')
    em.line(f"  {'Rule':<22}{'Period':<7}{'n':>7}{'lift':>8}{'p':>9}")
    rows = []
    # The min_pos rules are evaluated on the same reachable panel as every other
    # classifier, so that the lifts and the overlap below share a population.
    for rule, m0 in [('min_pos >= 0.98', minp >= 0.98), ('min_pos <= 0.05', minp <= 0.05)]:
        for k in ['Full', 'IS', 'OOS']:
            c = conditional_cell(fwd[m0 & masks[k]], base[k][0], base[k][1], vif)
            if c:
                em.line(f"  {rule:<22}{k:<7}{c['n']:>7}{c['lift']:>+8.1f}{c['p']:>9.4f}")
                rows.append([rule, k, c['n'], round(c['lift'], 1), round(c['p'], 4)])
    em.csv('table12b_minpos_classifier.csv', ['rule', 'period', 'n', 'lift_pp', 'p'], rows)
    sw = set(np.where((Z[('GFC', norms[0])] <= THR) & valid)[0])
    sm = set(np.where((minp >= 0.98) & valid)[0])
    em.line(f'\n  Overlap of min_pos >= 0.98 with the GFC classifier: {len(sw & sm)} days '
            f'(expected under independence {len(sw)*len(sm)/valid.sum():.1f}; '
            f'Jaccard {len(sw & sm)/len(sw | sm):.3f})')

    # ══ Table 14: non-topological alternatives ══════════════════════════
    em.head('Table 14  Non-topological alternatives: realised volatility')
    rv60 = realised_vol(ph, idx_h, 60)
    rv20 = realised_vol(ph, idx_h, 20)
    em.line('  The Takens point cloud is centred and scaled by its own scalar standard')
    em.line('  deviation before homology is computed, so the amplitude of the price path')
    em.line('  is divided out. S_t therefore describes the shape of the normalised')
    em.line('  trajectory rather than its magnitude.\n')
    em.line(f"  {'Diagram statistic':<22}{'Volatility window':<22}{'Pearson':>10}{'Spearman':>11}")
    em.line('  ' + '-' * 66)
    rows = []
    for dn, dv in [('S_t', S), ('sqrt(S_t)', np.sqrt(S)), ('maximum persistence', maxp)]:
        for vn, vv in [('HYG 60-day', rv60), ('HYG 20-day', rv20)]:
            pr, sr = pearsonr(dv, vv)[0], spearmanr(dv, vv).statistic
            em.line(f'  {dn:<22}{vn:<22}{pr:>+10.3f}{sr:>+11.3f}')
            rows.append([dn, vn, round(pr, 3), round(sr, 3)])
    em.csv('table14_volatility_correlation.csv',
           ['diagram_statistic', 'volatility_window', 'pearson', 'spearman'], rows)

    em.line('\n  Volatility-based classifiers (252-day rolling Z-score, z <= -1.5):')
    em.line(f"  {'Classifier':<26}" + ''.join([f"{'n':>7}{'lift':>9}" for _ in range(3)]))
    em.line(f"  {'':<26}" + ''.join([f'{p:^16}' for p in ['Full', 'IS', 'OOS']]))
    em.line('  ' + '-' * 76)
    rows, vol_sets = [], {}
    for vn, vv in [('HYG 60-day realised vol', rv60), ('HYG 20-day realised vol', rv20)]:
        zv = rolling_z(vv)
        ev = evaluate(zv, fwd, masks, base, vif)
        # Two decimals: at one decimal the 20-day OOS cell prints as +0.0, which
        # reads as an exact zero rather than as a rounded small number.
        seg = ''.join([(f"{ev[k]['n']:>7}{ev[k]['lift']:>+9.2f}" if ev[k]
                        else f"{'--':>7}{'--':>9}") for k in ['Full', 'IS', 'OOS']])
        em.line(f'  {vn:<26}{seg}')
        rows.append([vn] + [x for k in ['Full', 'IS', 'OOS']
                            for x in ([ev[k]['n'], round(ev[k]['lift'], 2)] if ev[k] else ['', ''])])
        vol_sets[vn] = set(np.where((zv <= THR) & ~np.isnan(zv))[0])
    em.csv('table14b_volatility_classifier.csv',
           ['classifier'] + [f'{p}_{s}' for p in ['full', 'is', 'oos'] for s in ['n', 'lift']], rows)

    em.line('\n  Overlap of the topological classifiers with the volatility classifiers:')
    em.line(f"  {'Topological classifier':<28}{'Volatility classifier':<26}"
            f"{'common':>8}{'expected':>10}{'Jaccard':>9}")
    em.line('  ' + '-' * 84)
    rows = []
    topo = [(f'W2 GFC / {nname[c]}', Z[('GFC', c)]) for c in norms]
    topo += [('maximum persistence', Zfree['maximum persistence'])]
    for tn, tz in topo:
        ts = set(np.where((tz <= THR) & ~np.isnan(tz))[0])
        for vn, vs in vol_sets.items():
            nv2 = int((~np.isnan(tz)).sum())
            inter, expd = len(ts & vs), len(ts) * len(vs) / nv2
            jac = inter / len(ts | vs)
            em.line(f'  {tn:<28}{vn:<26}{inter:>8}{expd:>10.1f}{jac:>9.3f}')
            rows.append([tn, vn, inter, round(expd, 1), round(jac, 3)])
    em.csv('table14c_volatility_overlap.csv',
           ['topological_classifier', 'volatility_classifier', 'common_days',
            'expected_under_independence', 'jaccard'], rows)

    # ══ Table 16: path efficiency ═══════════════════════════════════════
    em.head('Table 16  What a low S_t describes: path efficiency of the 60-day window')
    em.line('  E_t = |net displacement| / (total distance travelled) over the Takens window.')
    em.line('  1.0 for a monotone path, near 0 for a path that goes out and comes back.\n')
    eok = ~np.isnan(effr)
    em.line(f'  n = {int(eok.sum())}')
    em.line(f'  Pearson  corr(S_t, E_t)          = {pearsonr(S[eok], effr[eok])[0]:+.3f}')
    em.line(f'  Spearman corr(S_t, E_t)          = {spearmanr(S[eok], effr[eok]).statistic:+.3f}')
    em.line(f'  Pearson  corr(S_t, dir. changes) = {pearsonr(S[eok], dchg[eok])[0]:+.3f}')
    em.line(f'  Spearman corr(S_t, dir. changes) = {spearmanr(S[eok], dchg[eok]).statistic:+.3f}')
    em.line(f'  mean direction changes           = {dchg[eok].mean():.1f} '
            f'(sd {dchg[eok].std():.1f}, out of at most {W - 2})')
    em.line(f"\n  {'Decile of S_t':<16}{'mean S_t':>12}{'mean E_t':>11}{'dir. changes':>15}")
    em.line('  ' + '-' * 54)
    rows = []
    qe = np.quantile(S[eok], np.linspace(0, 1, 11))
    for i in range(10):
        m = eok & (S >= qe[i]) & ((S <= qe[i + 1]) if i == 9 else (S < qe[i + 1]))
        em.line(f'  {i+1:<16}{S[m].mean():>12.4f}{effr[m].mean():>11.3f}{dchg[m].mean():>15.1f}')
        rows.append([i + 1, round(float(S[m].mean()), 4), round(float(effr[m].mean()), 3),
                     round(float(dchg[m].mean()), 1)])
    em.csv('table16_path_efficiency_deciles.csv',
           ['decile', 'S_mean', 'efficiency_ratio_mean', 'dir_changes_mean'], rows)
    fig7_dec = ([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])

    em.line(f"\n  {'Day set':<30}{'n':>7}{'mean E_t':>11}{'median E_t':>13}")
    em.line('  ' + '-' * 61)
    rows = []
    for lab, m in ([('all days', eok)] +
                   [(f'classified / {nname[cnt]}', eok & (Z[('GFC', cnt)] <= THR))
                    for cnt in norms]):
        em.line(f'  {lab:<30}{int(m.sum()):>7}{effr[m].mean():>11.3f}'
                f'{np.median(effr[m]):>13.3f}')
        rows.append([lab, int(m.sum()), round(float(effr[m].mean()), 3),
                     round(float(np.median(effr[m])), 3)])
    em.csv('table16b_path_efficiency_classified.csv',
           ['day_set', 'n', 'efficiency_ratio_mean', 'efficiency_ratio_median'], rows)
    fig7_sets = [('all days', effr[eok])] + \
                [(nname[cnt], effr[eok & (Z[('GFC', cnt)] <= THR)]) for cnt in norms]
    em.line('\n  The two constructions select opposite ends of the same descriptive axis.')

    # ══ Table 13: Z-score denominator ═══════════════════════════════════
    em.head('Table 13  Behaviour of the rolling Z-score denominator')
    zg, sig = rolling_z(DIST[('GFC', norms[0])], return_sigma=True)
    sv = ~np.isnan(sig)
    pct = np.full(len(sig), np.nan)
    pct[sv] = [float(np.mean(sig[sv] <= s) * 100) for s in sig[sv]]
    msel = (zg <= THR) & ~np.isnan(zg)
    em.line(f'  sigma_w,252 over all days: median {np.nanmedian(sig):.4f}  '
            f'[p10 {np.nanpercentile(sig,10):.4f}, p90 {np.nanpercentile(sig,90):.4f}]')
    em.line(f'  sigma on classified days:  median {np.nanmedian(sig[msel]):.4f}  '
            f'= {np.nanmedian(pct[msel]):.1f}th percentile of that distribution')
    em.line(f'  classified days in the lowest sigma quintile: '
            f'{np.mean(pct[msel] <= 20)*100:.1f}%  (mechanical expectation 20.0%)')
    em.line(f'  raw distance w_t: classified median {np.nanmedian(DIST[("GFC",norms[0])][msel]):.4f} '
            f'vs all days {np.nanmedian(DIST[("GFC",norms[0])][sv]):.4f}')
    em.csv('table13_zscore_denominator.csv',
           ['quantity', 'value'],
           [['sigma_median_all', round(float(np.nanmedian(sig)), 5)],
            ['sigma_median_classified', round(float(np.nanmedian(sig[msel])), 5)],
            ['sigma_percentile_classified', round(float(np.nanmedian(pct[msel])), 1)],
            ['pct_in_lowest_quintile', round(float(np.mean(pct[msel] <= 20) * 100), 1)],
            ['raw_distance_median_classified',
             round(float(np.nanmedian(DIST[('GFC', norms[0])][msel])), 5)],
            ['raw_distance_median_all',
             round(float(np.nanmedian(DIST[('GFC', norms[0])][sv])), 5)]])

    # ══ Table 15: threshold marginality ═════════════════════════════════
    em.head('Table 15  Threshold marginality')
    em.line('  How many days sit within a given distance of the classifier threshold,')
    em.line('  and how many classified days a perturbation of that size could move.\n')
    em.line(f"  {'Construction':<20}{'|z - thr| <':<14}{'days':>8}{'as % of':>10}")
    em.line(f"  {'':<20}{'':<14}{'':>8}{'classified':>10}")
    em.line('  ' + '-' * 56)
    rows = []
    for cnt in norms:
        zc = Z[('GFC', cnt)]
        vmask = ~np.isnan(zc)
        ncl = int(((zc <= THR) & vmask).sum())
        for eps in (0.001, 0.005, 0.01, 0.05):
            k = int((np.abs(zc[vmask] - THR) < eps).sum())
            em.line(f'  {nname[cnt]:<20}{eps:<14.3f}{k:>8}{100*k/ncl:>9.1f}%')
            rows.append([nname[cnt], eps, k, ncl, round(100 * k / ncl, 1)])
        em.line('')
    em.csv('table15_threshold_marginality.csv',
           ['construction', 'epsilon', 'days_within_epsilon', 'n_classified',
            'pct_of_classified'], rows)
    em.line('  A day whose Z-score lies within epsilon of the threshold is decided by')
    em.line('  perturbations of that size in the input series.')

    # ══ Table 17: drawdown stratification of the OOS classified days ════
    em.head('Table 17  Out-of-sample classified days by S&P 500 drawdown from the '
            '252-day peak')
    sp_dates = [d for d in dates_h if d in spx]
    dd_map = dict(zip(sp_dates, drawdown_from_peak([spx[d] for d in sp_dates])))
    DD = np.array([dd_map.get(d, np.nan) for d in dates_h])
    ddok = ~np.isnan(DD)
    oos = masks['OOS']
    buckets = [('>= -2%', DD >= -2),
               ('-2% to -5%', (DD < -2) & (DD >= -5)),
               ('-5% to -10%', (DD < -5) & (DD >= -10)),
               ('< -10%', DD < -10)]

    em.line('\n  Cumulative S&P 500 return over each panel:')
    cum_rows = []
    for lab, m in [('IS panel', masks['IS']), ('OOS panel', oos)]:
        ds = [d for d, k in zip(dates_h, m) if k and d in spx]
        c = (spx[ds[-1]] / spx[ds[0]] - 1) * 100
        em.line(f'    {lab:<12}{ds[0]} .. {ds[-1]}   {c:>+8.1f}%')
        cum_rows.append([lab, ds[0], ds[-1], round(c, 1)])
    em.csv('table17b_panel_cumulative_return.csv',
           ['panel', 'start', 'end', 'cum_return_pct'], cum_rows)

    em.line(f"\n  {'Bucket':<14}{'panel n':>9}{'panel %':>9}{'uncond':>9}"
            + ''.join(f'{nname[c][:8] + " n":>12}{"% of cls":>10}{"lift":>8}'
                      for c in norms))
    em.line('  ' + '-' * (41 + 30 * len(norms)))
    rows = []
    for lab, sel in buckets:
        sel = sel & ddok
        npan = int((oos & sel).sum())
        ubase = float(np.mean(fwd[oos & sel] > 0))
        line = (f'  {lab:<14}{npan:>9}{npan/int(oos.sum())*100:>8.1f}%'
                f'{ubase*100:>8.1f}%')
        row = [lab, npan, round(npan / int(oos.sum()) * 100, 1), round(ubase * 100, 1)]
        for cnt in norms:
            cls = oos & sel & (Z[('GFC', cnt)] <= THR)
            ncls = int(cls.sum())
            ntot = int((oos & (Z[('GFC', cnt)] <= THR)).sum())
            share = ncls / ntot * 100 if ntot else float('nan')
            lift = ((np.mean(fwd[cls] > 0) - ubase) * 100) if ncls else float('nan')
            line += f'{ncls:>12}{share:>9.1f}%{lift:>+8.1f}'
            row += [ncls, round(share, 1), None if ncls == 0 else round(float(lift), 1)]
        em.line(line)
        rows.append(row)
    hdr = ['bucket', 'oos_panel_n', 'oos_panel_pct', 'uncond_pos_freq_pct']
    for cnt in norms:
        t = 'counting' if cnt else 'published'
        hdr += [f'{t}_n', f'{t}_pct_of_classified', f'{t}_lift_pp']
    em.csv('table17_drawdown_strata.csv', hdr, rows)
    fig8_labels = [r[0] for r in rows]
    fig8_panel = [r[2] for r in rows]
    fig8_cls = [[r[5 + 3 * j] for r in rows] for j in range(len(norms))]
    em.line('\n  Lift is against the unconditional frequency within the same bucket.')
    em.line('  Bucket counts are small; no tail probability is attached.')

    # ══ Table 18: activation episodes and an alternative outcome ════════
    em.head('Table 18  Activation clustering, and the outcome variable')
    em.line('  A classified-day count treats clustered activations as independent')
    em.line(f'  draws. Episodes below are runs separated by more than {HOR} trading days.\n')
    em.line(f"  {'Classifier':<26}{'Panel':<7}{'n':>5}{'Episodes':>10}"
            f"{'Largest':>9}{'Eff. N':>8}")
    em.line('  ' + '-' * 66)
    rows = []
    for cnt in norms:
        zc = Z[('GFC', cnt)]
        for pk in ('Full', 'OOS'):
            sel = (zc <= THR) & masks[pk]
            eps = activation_episodes(sel)
            if not eps:
                continue
            big = max(len(e) for e in eps)
            em.line(f'  GFC / {nname[cnt]:<19}{pk:<7}{int(sel.sum()):>5}{len(eps):>10}'
                    f'{big:>9}{sel.sum()/vif:>8.1f}')
            rows.append([f'GFC / {nname[cnt]}', pk, int(sel.sum()), len(eps), big,
                         round(float(sel.sum() / vif), 1)])
        eps = activation_episodes((zc <= THR) & masks['OOS'])
        em.line('     OOS episodes: ' + ', '.join(
            f'{dates_h[e[0]]} (n={len(e)})' for e in eps))
        em.line('')
    em.csv('table18_activation_episodes.csv',
           ['classifier', 'panel', 'n_days', 'n_episodes', 'largest_episode', 'eff_n'],
           rows)

    hyg_fwd = forward_returns(dates_h, {d: p for d, p in zip(dh, ph)})
    hok = ~np.isnan(hyg_fwd) & masks['Full']
    em.line('  The outcome throughout is the 20-day forward S&P 500 return, following')
    em.line('  Takagi (2026a). For comparison, the same classified days measured')
    em.line("  against HYG's own 20-day forward return:\n")
    em.line(f"  {'Day set':<30}{'n':>6}{'Pos. freq.':>12}{'Mean':>10}")
    em.line('  ' + '-' * 58)
    rows = [['all reachable days', int(hok.sum()),
             round(float(np.mean(hyg_fwd[hok] > 0) * 100), 2),
             round(float(np.mean(hyg_fwd[hok])), 3)]]
    em.line(f"  {'all reachable days':<30}{int(hok.sum()):>6}"
            f'{np.mean(hyg_fwd[hok]>0)*100:>11.1f}%{np.mean(hyg_fwd[hok]):>+9.2f}%')
    for cnt in norms:
        s = hok & (Z[('GFC', cnt)] <= THR)
        em.line(f'  classified / {nname[cnt]:<17}{int(s.sum()):>6}'
                f'{np.mean(hyg_fwd[s]>0)*100:>11.1f}%{np.mean(hyg_fwd[s]):>+9.2f}%')
        rows.append([f'classified / {nname[cnt]}', int(s.sum()),
                     round(float(np.mean(hyg_fwd[s] > 0) * 100), 2),
                     round(float(np.mean(hyg_fwd[s])), 3)])
    em.csv('table18b_hyg_own_outcome.csv',
           ['day_set', 'n', 'pos_freq_pct', 'mean_pct'], rows)
    em.line('\n  Reported for scope, not as a second test: the base rate, the')
    em.line('  dependence structure and the multiplicity all change with the outcome.')

    # ══ Table 19: the same composition in a non-topological indicator ═══
    em.head('Table 19  The same composition applied to a Mahalanobis turbulence index')
    em.line('  Kritzman and Li (2010) measure turbulence as the Mahalanobis distance of the')
    em.line('  current cross-section of returns from a reference mean and covariance. It is')
    em.line('  a distance to a historical reference, standardised over a trailing window and')
    em.line('  thresholded -- the composition this paper is about, with no topology in it.')
    em.line('  Below: how much of the thresholded quantity survives discarding the reference.\n')
    hyg_map = dict(zip(dh, ph))
    jnk_map = dict(zip(dj, pj))
    tdates = [d for d in dates_h if d in hyg_map and d in jnk_map and d in spx]
    Pm = np.array([[hyg_map[d], jnk_map[d], spx[d]] for d in tdates])
    rt = np.diff(np.log(Pm), axis=0)
    rdates = tdates[1:]
    refs = [('full-sample', rt.mean(0), np.cov(rt.T))]
    gfc = REF[('HYG', 'GFC')][1]
    if gfc in rdates and rdates.index(gfc) >= W:
        k = rdates.index(gfc)
        wnd = rt[k - W:k]
        refs.append((f'crisis window to {gfc}', wnd.mean(0), np.cov(wnd.T)))
    rows = []
    for lab, tmu, tcov in refs:
        D2, quad, plain, const = mahalanobis_turbulence(rt, tmu, tcov)
        zD, zq, zp = rolling_z(D2), rolling_z(quad), rolling_z(plain)
        f = ~np.isnan(zD) & ~np.isnan(zq) & ~np.isnan(zp)
        em.line(f'  Reference: {lab}')
        em.line(f"    additive constant mu' S^-1 mu = {const:.3e}, "
                f'removed exactly by the Z-score')
        em.line(f"    {'Reference discarded':<28}{'corr, levels':>14}{'corr, Z-scored':>16}"
                f"{'Jaccard':>10}")
        em.line('    ' + '-' * 68)
        for dlab, v, z in [('mean only', quad, zq), ('mean and covariance', plain, zp)]:
            A = set(np.where(f & (zD >= -THR))[0])
            B = set(np.where(f & (z >= -THR))[0])
            j = len(A & B) / len(A | B) if (A | B) else float('nan')
            rl, rz = pearsonr(D2, v)[0], pearsonr(zD[f], z[f])[0]
            em.line(f'    {dlab:<28}{rl:>+14.4f}{rz:>+16.4f}{j:>10.3f}')
            rows.append([lab, dlab, round(float(rl), 4), round(float(rz), 4),
                         len(A), len(B), len(A & B), round(float(j), 3)])
        em.line('')
    em.csv('table19_turbulence_analogue.csv',
           ['reference', 'discarded', 'corr_levels', 'corr_zscored',
            'n_full', 'n_reference_free', 'n_common', 'jaccard'], rows)
    em.line('  The classifier here is z >= +1.5, the upper tail, since turbulence is')
    em.line('  extreme when large. Reported as a scope check on the composition, not as a')
    em.line('  finding about turbulence: the outcome variable, the base rate and the')
    em.line('  dependence structure of a turbulence classifier are all outside this paper.')

    # ══ figures ═════════════════════════════════════════════════════════
    if not args.no_figures:
        em.head('Figures')
        plt = setup_mpl()
        s_ref = {nm: diagram_stats(REF[('HYG', nm)][0])['sq_pers'] for nm in NADIRS}
        figure_distance_content(plt, args.outdir, DIST, S, ncard, s_ref)
        if len(norms) == 2:
            figure_structure(plt, args.outdir, full_stats, cls_stats[False], cls_stats[True])
        figure_reference_sensitivity(plt, args.outdir, fig3rows)
        figure_free_and_overlap(plt, args.outdir, fig4rows, jl, fcols, np.asarray(jm))
        if fig5panels:
            figure_shift_null(plt, args.outdir, fig5panels[:6], base['Full'][0])
        refpts = []
        for nm in NADIRS:
            dg, used = REF[('HYG', nm)]
            st = diagram_stats(dg)
            refpts.append((f'HYG {nm}', PD['HYG'][dh.index(used)]['min_pos'],
                           st['loops'], st['entropy']))
        figure_minpos(plt, args.outdir, centres, mloops, ment, refpts)
        figure_efficiency(plt, args.outdir, S, effr,
                          fig7_dec[0], fig7_dec[1], fig7_dec[2], fig7_sets)
        figure_drawdown(plt, args.outdir, fig8_labels, fig8_panel, fig8_cls,
                        [nname[c] for c in norms])

    em.save()
    print(f'\n  tables  -> {os.path.join(args.outdir, "tables")}')
    if not args.no_figures:
        print(f'  figures -> {os.path.join(args.outdir, "figures")}')
    print('  done.')


if __name__ == '__main__':
    main()
