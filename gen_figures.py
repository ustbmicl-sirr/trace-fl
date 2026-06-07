"""Generate ICDM data figures from TRACE 3-seed results (results_trace/*.csv).

All data figures are produced here (matplotlib), with a single shared style so
the figures are visually consistent: one serif font matching the paper, one
fixed colour per aggregation rule / method across every figure, uniform grid,
tick, and legend styling. The only non-matplotlib figure is the architecture
schematic (figures/architecture.tex), which is intentionally a TikZ diagram.

All numbers come from the TRACE engine (no MatSwarm / old-engine data).
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_trace")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# Shared style: one look for every data figure.
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": "0.8",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,
    "lines.linewidth": 1.7,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL = 3.4  # IEEE column width (in); generate at display size -> correct fonts

# one fixed colour per aggregation rule, used in every figure
GAR_COLOR = {
    "mean": "#7f7f7f", "centered_clipping": "#e6a02c", "median": "#2ca25f",
    "geo_med": "#66c2a4", "multi_krum": "#3182bd", "bulyan": "#08519c",
}
TARA_C, ORACLE_C, FLTRUST_C = "#6a51a3", "#2ca25f", "#de2d26"
CLEAN_C, ATTACK_C = "#2ca25f", "#de2d26"

GAR_LABEL = {"mean": "Mean", "median": "Median", "multi_krum": "Multi-Krum",
             "centered_clipping": "Cent.Clip", "bulyan": "Bulyan",
             "geo_med": "Geo.Med"}
ATK_LABEL = {"sign_flip": "sign-flip", "scale": "scale", "gaussian": "Gaussian",
             "little_is_enough": "LIE", "fall_of_empires": "FoE"}
GAR_ORDER = ["mean", "centered_clipping", "median", "geo_med", "multi_krum", "bulyan"]
ATK_ORDER = ["sign_flip", "scale", "gaussian", "little_is_enough", "fall_of_empires"]


def _read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def _bars(ax, x, series, width):
    """series: list of (label, values, color). Adds thin edges, uniform look."""
    k = len(series)
    offs = (np.arange(k) - (k - 1) / 2) * width
    for (label, vals, color), off in zip(series, offs):
        ax.bar(x + off, vals, width, label=label, color=color,
               edgecolor="0.25", linewidth=0.4)
    ax.axhline(0, color="0.4", lw=0.6, zorder=0)


# ----------------------------------------------------------------------------
def robustness_heatmap():
    rows = _read("robustness_n7_f1_iid.csv")
    table = {(r["gar"], r["attack"]): _f(r["attack_r2"]) for r in rows}
    M = np.array([[table.get((g, a), np.nan) for a in ATK_ORDER] for g in GAR_ORDER])
    disp = np.clip(np.nan_to_num(M, nan=-1.0), -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.grid(False)
    im = ax.imshow(disp, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATK_ORDER)))
    ax.set_xticklabels([ATK_LABEL[a] for a in ATK_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(GAR_ORDER)))
    ax.set_yticklabels([GAR_LABEL[g] for g in GAR_ORDER])
    for i in range(len(GAR_ORDER)):
        for j in range(len(ATK_ORDER)):
            v = M[i, j]
            txt = r"$\downarrow$" if (math.isnan(v) or v < -1) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("test $R^2$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(r"Robustness by rule $\times$ attack ($n{=}7$, $f{=}1$)")
    _save(fig, "robustness_heatmap.pdf")


def multiscale_bars():
    rows = sorted([r for r in _read("multiscale_f1_sign_flip_iid.csv")
                   if r["trust"] == "byzantine"], key=lambda r: int(r["n"]))
    x = np.arange(len(rows))
    labels = [f"$n{{=}}{r['n']}$\n{GAR_LABEL[r['gar']]}" for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    _bars(ax, x, [("clean", [_f(r["clean_r2"]) for r in rows], CLEAN_C),
                  ("sign-flip", [_f(r["attack_r2"]) for r in rows], ATTACK_C)], 0.38)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("test $R^2$")
    ax.set_title("Resolved rule across scale (byzantine, $f{=}1$)")
    ax.legend(frameon=False)
    _save(fig, "multiscale_bars.pdf")


def substitution_bars():
    order = {g: i for i, g in enumerate(GAR_ORDER)}
    rows = sorted(_read("substitution_n7_f1_sign_flip_iid.csv"),
                  key=lambda r: order.get(r["gar"], 99))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    _bars(ax, x, [("clean", [_f(r["clean_r2"]) for r in rows], CLEAN_C),
                  ("sign-flip", [_f(r["attack_r2"]) for r in rows], ATTACK_C)], 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([GAR_LABEL[r["gar"]] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("test $R^2$")
    ax.set_title("Same-context substitution ($n{=}7$, $f{=}1$, sign-flip)")
    ax.legend(frameon=False)
    _save(fig, "substitution_bars.pdf")


def tara_bars():
    order = {a: i for i, a in enumerate(ATK_ORDER)}
    rows = sorted([r for r in _read("adaptive_n7_iid.csv") if r["true_f"] == "2"],
                  key=lambda r: order.get(r["attack"], 99))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.05, 2.45))
    _bars(ax, x, [("TARA", [_f(r["tara_r2"]) for r in rows], TARA_C),
                  ("oracle-$f$", [_f(r["oracle_r2"]) for r in rows], ORACLE_C),
                  ("FLTrust", [max(_f(r["fltrust_r2"]), -0.05) for r in rows], FLTRUST_C)],
          0.27)
    ax.set_xticks(x)
    ax.set_xticklabels([ATK_LABEL[r["attack"]] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("test $R^2$")
    ax.set_title("$n{=}7$, $f{=}2$")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "tara_bars.pdf")


def fhat_accuracy():
    rows = _read("adaptive_n7_iid.csv")
    by_f = {}
    for r in rows:
        by_f.setdefault(int(r["true_f"]), []).append(_f(r["fhat_mean"]))
    tf = sorted(by_f)
    fh = [sum(by_f[k]) / len(by_f[k]) for k in tf]
    fig, ax = plt.subplots(figsize=(1.35, 2.45))
    ax.plot(tf, tf, "--", color="0.5", label=r"ideal")
    ax.plot(tf, fh, "o-", color=TARA_C, label=r"$\hat f$")
    ax.set_xlabel("true $f$"); ax.set_ylabel(r"est.\ $\hat f$")
    ax.set_xticks(tf); ax.set_yticks(tf)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Estimation")
    _save(fig, "fhat_accuracy.pdf")


def benchmark_fig():
    import pandas as pd
    base = os.path.join(ROOT, "data", "optimade_processed")
    srcs = [("OQMD", "oqmd", GAR_COLOR["multi_krum"]),
            ("Mat.Proj.", "mp", GAR_COLOR["median"]),
            ("Alexandria", "alexandria", FLTRUST_C)]
    data = {label: (pd.read_csv(os.path.join(base, f"{k}_featurized.csv")), c)
            for label, k, c in srcs}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL, 1.95),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    for label, (df, c) in data.items():
        y = df["e_form"].dropna()
        y = y[(y > -5) & (y < 3)]
        ax1.hist(y, bins=40, density=True, histtype="step", lw=1.4, color=c, label=label)
    ax1.set_xlabel("$e_\\mathrm{form}$ (eV/atom)"); ax1.set_ylabel("density")
    ax1.set_title("Label shift")
    ax1.legend(frameon=False)
    labels = list(data.keys())
    sets = {k: set(v[0]["formula"].astype(str)) for k, v in data.items()}
    n = len(labels)
    M = np.array([[100.0 if i == j else
                   100.0 * len(sets[labels[i]] & sets[labels[j]]) /
                   max(1, min(len(sets[labels[i]]), len(sets[labels[j]])))
                   for j in range(n)] for i in range(n)])
    ax2.grid(False)
    im = ax2.imshow(M, cmap="Blues", vmin=0, vmax=100)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
    ax2.set_yticklabels(labels, fontsize=6.5)
    ax2.yaxis.tick_right()           # row labels on the right to avoid crowding
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                     fontsize=6.5, color="black" if M[i, j] < 50 else "white")
    ax2.set_title("Overlap (%)")
    _save(fig, "benchmark_char.pdf")


def adv_fig():
    rows = _read("adv_sweep_n7_iid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.0), sharey=True)
    for atk, xlabel, ax, logx in [
            ("little_is_enough", "LIE strength $z$", axes[0], False),
            ("scale", "scale factor", axes[1], True)]:
        rs = sorted([r for r in rows if r["attack"] == atk],
                    key=lambda r: float(r["strength"]))
        x = [float(r["strength"]) for r in rs]
        ax.plot(x, [_f(r["tara_r2"]) for r in rs], "o-", color=TARA_C,
                lw=2.0, ms=4, zorder=3, label="TARA ($f$ hidden)")
        ax.plot(x, [_f(r["bulyan_r2"]) for r in rs], "s--", color=GAR_COLOR["multi_krum"],
                lw=1.4, ms=3.5, label="Multi-Krum (oracle $f$)")
        ax.plot(x, [_f(r["median_r2"]) for r in rs], "^:", color=GAR_COLOR["median"],
                lw=1.4, ms=3.5, label="Median")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("test $R^2$"); axes[0].set_ylim(0.40, 0.66)
    axes[0].annotate(r"flat: $\hat f{=}2$", xy=(2.0, 0.594), xytext=(0.55, 0.45),
                     fontsize=6.5, color=TARA_C,
                     arrowprops=dict(arrowstyle="->", color=TARA_C, lw=0.7))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), fontsize=6.8, columnspacing=1.0)
    _save(fig, "adv_sweep.pdf")


def convergence_fig():
    rows = _read("convergence_n7_iid.csv")
    series = {}
    for r in rows:
        series.setdefault(r["series"], []).append((int(r["round"]), _f(r["r2"])))
    style = {"Bulyan (clean)": (GAR_COLOR["bulyan"], "-"),
             "Bulyan (sign-flip)": (GAR_COLOR["bulyan"], "--"),
             "TARA (sign-flip)": (TARA_C, "-"),
             "Mean (sign-flip)": (GAR_COLOR["mean"], ":")}
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for name, pts in series.items():
        pts.sort()
        c, ls = style.get(name, ("gray", "-"))
        ax.plot([p[0] for p in pts], [max(p[1], -0.3) for p in pts],
                ls, color=c, lw=1.7, label=name)
    ax.set_xlabel("global round"); ax.set_ylabel("test $R^2$")
    ax.set_ylim(-0.3, 0.7)
    ax.set_title("Convergence ($n{=}7$, $f{=}1$)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "convergence.pdf")


if __name__ == "__main__":
    robustness_heatmap()
    multiscale_bars()
    substitution_bars()
    tara_bars()
    fhat_accuracy()
    benchmark_fig()
    adv_fig()
    convergence_fig()
    print("done ->", FIG)
