#!/usr/bin/env python3
"""
plot_dm_exgal_vs_z.py
---------------------
Scatter-plot extragalactic DM against redshift for the FRB catalogue:
    x-axis = DM_exgal (pc cm^-3)
    y-axis = redshift z
with axis labels (and units), a title, and an optional linear trend line.

Reads the CSV produced by frb_dm_exgal_ne2025.py (the one that has the
"DM exgal (pc cm^-3)" and "z" columns).

Run
---
    python plot_dm_exgal_vs_z.py dm_ex_vs_z_filled_ne2025.csv
    python plot_dm_exgal_vs_z.py dm_ex_vs_z_filled_ne2025.csv my_plot.png

Needs: numpy, pandas, matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- CONFIG ------------------------------------- #

ADD_TREND     = True       # draw a least-squares linear trend line (z vs DM)
DROP_NEGATIVE = False      # if True, exclude bursts with DM_exgal < 0 from the plot
SAVE_DPI      = 200
SHOW_WINDOW   = True        # pop up an interactive window after saving

MISSING = {"", "..", "...", "nan", "n/a", "na", "-"}


# ------------------------------- helpers ------------------------------------ #

def as_float(v):
    if v is None or str(v).strip().lower() in MISSING:
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def find_col(cols, *cands):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


# --------------------------------- main ------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Plot DM_exgal vs redshift for FRBs.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    args = ap.parse_args()
    out_png = args.output or args.input.rsplit(".", 1)[0] + "_dm_vs_z.png"

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    cols = list(df.columns)

    c_dm = find_col(cols, "DM exgal (pc cm^-3)", "dm exgal", "dm_exgal", "dm_ex")
    c_z  = find_col(cols, "z", "redshift")
    if not (c_dm and c_z):
        raise SystemExit(
            f"Couldn't find the needed columns. DM_exgal={c_dm!r}, z={c_z!r}. "
            "Did you run frb_dm_exgal_ne2025.py first?")

    # collect rows where BOTH values are present and numeric
    x, y = [], []
    for i in range(len(df)):
        dm = as_float(df.at[i, c_dm])
        z  = as_float(df.at[i, c_z])
        if dm is None or z is None:
            continue
        if DROP_NEGATIVE and dm < 0:
            continue
        x.append(dm)
        y.append(z)
    x = np.array(x)
    y = np.array(y)
    if len(x) == 0:
        raise SystemExit("No bursts have both DM_exgal and z -- nothing to plot.")

    # ---- plot ---- #
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=40, alpha=0.75, edgecolor="black", linewidth=0.5,
               color="#3b6ea5", label=f"FRBs (n = {len(x)})", zorder=3)

    if ADD_TREND and len(x) >= 2:
        m, b = np.polyfit(x, y, 1)                    # z = m*DM + b
        xline = np.linspace(x.min(), x.max(), 200)
        # R^2 of the fit
        resid = y - (m * x + b)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.plot(xline, m * xline + b, color="crimson", lw=2, zorder=2,
                label=f"linear fit: z = {m:.2e}·DM + {b:.3f}  (R² = {r2:.2f})")

    ax.set_xlabel(r"Extragalactic DM,  $\mathrm{DM_{ex\,gal}}$  (pc cm$^{-3}$)")
    ax.set_ylabel(r"Redshift,  $z$")
    ax.set_title("Extragalactic Dispersion Measure vs. Redshift for FRBs")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(frameon=True)
    fig.tight_layout()

    fig.savefig(out_png, dpi=SAVE_DPI)
    print(f"Saved {out_png}  ({len(x)} bursts plotted)")

    if SHOW_WINDOW:
        plt.show()


if __name__ == "__main__":
    main()