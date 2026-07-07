"""
Plot FRB scattering timescale (tau, ms) versus dispersion measure (DM, pc cm^-3).

  * DM on the x-axis, tau on the y-axis.
  * Detections are drawn as points with asymmetric vertical error bars
    (independent + and - errors).
  * Upper limits -- any tau value written in the CSV as "<value"
    (e.g. "< 0.15" or "<0.4") -- are drawn as downward arrows instead of
    points, with the tail of the arrow sitting at the limiting value.
  * Optionally, points and arrows are colour-coded by the frequency (GHz) at
    which the scattering was measured (set FREQ_COL to enable). This is useful
    because tau is strongly frequency-dependent.

Only the CONFIG block below should need editing. Set the column names to match
your catalogue, then run:  python plot_tau_vs_dm.py

Extra columns in the file are ignored -- only the columns named below are used.
Rows with a missing/unreadable DM or tau are skipped and counted.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================== CONFIG ===============================
CSV_PATH = "frb data and graphs (nexus 2026, chatterjee & cordes) - graph_tau_dm.csv"          # path to your file

# This sheet has a SINGLE header row (frb_name, tau_scatt, ...), so header=0
# keeps row 0 as the column names and we skip nothing. If you ever export a
# version with a second units row beneath the header, set SKIP_ROWS = [1].
HEADER_ROW = 0
SKIP_ROWS  = None

NA_VALUES  = [".."]                       # missing-value placeholder in the file

# --- Column names (exactly as they appear in the first header row) ---------
DM_COL     = "dm_exgal"                    # dispersion measure, pc cm^-3
TAU_COL    = "tau_scatt"                   # scattering time, ms (may start "<")
TAU_ERR_HI = "tau_err_up"                  # +error on tau, ms  (blank/.. for limits)
TAU_ERR_LO = "tau_err_down"               # -error on tau, ms  (blank/.. for limits)
DM_ERR_COL = None                          # e.g. "dm_err" to draw x error bars, or None
FREQ_COL   = "scatt_freq"                  # frequency (GHz) tau was measured at;
                                           # set None to turn OFF colour-coding
NAME_COL   = "frb_name"                    # used only to name bursts in warnings;
                                           # set None to skip

# --- Appearance ------------------------------------------------------------
LOG_Y      = True    # tau usually spans orders of magnitude -> log is clearer
LOG_X      = False   # DM is usually fine linear; set True if you prefer log
ARROW_FRAC = 0.35    # downward-arrow length as a fraction of the tau value
CMAP       = "viridis"   # colormap used when FREQ_COL is set
FREQ_LOG   = False       # use a log colour scale for frequency

# On a LOG y-axis, a lower error bar with err_down >= tau reaches toward zero
# and shoots to the bottom of the plot. If this is set to a number N, the drawn
# lower error is capped so the bar extends at most N decades below the point
# (purely cosmetic -- the data is unchanged). Leave None to draw errors as-is.
CLIP_LOWER_ERR_DECADES = None

TITLE      = "FRB scattering time vs. dispersion measure"
OUT_PNG    = "tau_vs_dm.png"
DPI        = 200
NO_FREQ_COLOR = "0.6"    # grey used for points/arrows that have no frequency
# =====================================================================


def parse_tau(value):
    """Return (numeric_value, is_upper_limit) for one tau cell.

    Anything that can't be read as a number (blank, '..', or unexpected text
    like 'not_measured') becomes NaN, so that row is dropped later rather than
    crashing the run.
    """
    if pd.isna(value):
        return np.nan, False
    s = str(value).strip()
    is_upper = s.startswith("<")
    if is_upper:
        s = s[1:].strip()                      # strip the "<" and any space
    try:
        return float(s), is_upper
    except ValueError:
        return np.nan, False


def main():
    # ---- load ----
    df = pd.read_csv(CSV_PATH, header=HEADER_ROW, skiprows=SKIP_ROWS,
                     na_values=NA_VALUES)

    # ---- check the configured column names actually exist ----
    required = [DM_COL, TAU_COL, TAU_ERR_HI, TAU_ERR_LO]
    if DM_ERR_COL is not None:
        required.append(DM_ERR_COL)
    if FREQ_COL is not None:
        required.append(FREQ_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "These configured column names were not found in the CSV: "
            f"{missing}\nColumns actually present: {list(df.columns)}\n"
            "Fix the names in the CONFIG block to match your file."
        )

    # ---- parse tau column into value + upper-limit flag ----
    parsed = df[TAU_COL].apply(parse_tau)
    tau      = np.array([p[0] for p in parsed], dtype=float)
    is_upper = np.array([p[1] for p in parsed], dtype=bool)

    dm = pd.to_numeric(df[DM_COL], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df[TAU_ERR_HI], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df[TAU_ERR_LO], errors="coerce").to_numpy(dtype=float)
    hi = np.nan_to_num(hi, nan=0.0)            # missing error -> no bar (0)
    lo = np.nan_to_num(lo, nan=0.0)

    dm_err = None
    if DM_ERR_COL is not None:
        dm_err = pd.to_numeric(df[DM_ERR_COL], errors="coerce").to_numpy(dtype=float)
        dm_err = np.nan_to_num(dm_err, nan=0.0)

    freq = None
    if FREQ_COL is not None:
        freq = pd.to_numeric(df[FREQ_COL], errors="coerce").to_numpy(dtype=float)

    # ---- keep only rows with a usable DM and tau ----
    good = np.isfinite(tau) & np.isfinite(dm)
    if LOG_Y:
        good &= tau > 0
    if LOG_X:
        good &= dm > 0

    det = good & ~is_upper                     # detections
    ul  = good & is_upper                      # upper limits

    # ---- warn about lower error bars that would cross zero on a log axis ----
    names = (df[NAME_COL].astype(str).to_numpy()
             if (NAME_COL is not None and NAME_COL in df.columns)
             else np.array([f"row {i}" for i in range(len(df))]))
    lo_plot = lo.copy()                        # copy used only for DRAWING
    if LOG_Y:
        bad = det & (lo >= tau) & (tau > 0)
        if bad.any():
            print("WARNING: these detections have tau_err_down >= tau, so their "
                  "lower error bar runs to ~0 on a log axis:")
            for i in np.where(bad)[0]:
                frac = lo[i] / tau[i]
                print(f"  {names[i]}: tau={tau[i]:g} ms, tau_err_down={lo[i]:g} ms "
                      f"({frac:.1f}x tau), tau_err_up={hi[i]:g} ms, DM={dm[i]:g}")
            print("  -> check these rows for a typo/unit slip. To cap the bars "
                  "cosmetically, set CLIP_LOWER_ERR_DECADES (e.g. 2).")
        # optional cosmetic cap: bar bottom no lower than tau / 10**N
        if CLIP_LOWER_ERR_DECADES is not None:
            floor = tau * (1.0 - 10.0 ** (-CLIP_LOWER_ERR_DECADES))
            lo_plot = np.minimum(lo_plot, floor)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    if freq is None:
        # -------- simple two-colour scheme (no frequency info) --------
        ax.errorbar(dm[det], tau[det], yerr=[lo_plot[det], hi[det]],
                    xerr=(dm_err[det] if dm_err is not None else None),
                    fmt="o", ms=6, color="#1f6feb", ecolor="#1f6feb",
                    capsize=3, elinewidth=1.1, label="Detection")
        ax.errorbar(dm[ul], tau[ul],
                    yerr=ARROW_FRAC * tau[ul], uplims=True,
                    xerr=(dm_err[ul] if dm_err is not None else None),
                    fmt="none", ecolor="#d1495b", elinewidth=1.4,
                    capsize=0, label="Upper limit")
        ax.legend(frameon=False)

    else:
        # -------- colour-coded by scattering frequency --------
        valid_f = good & np.isfinite(freq)
        if valid_f.any():
            fmin, fmax = np.nanmin(freq[valid_f]), np.nanmax(freq[valid_f])
        else:
            fmin, fmax = 0.0, 1.0
        if FREQ_LOG and fmin > 0:
            norm = mpl.colors.LogNorm(vmin=fmin, vmax=fmax)
        else:
            norm = mpl.colors.Normalize(vmin=fmin, vmax=fmax)
        cmap = plt.get_cmap(CMAP)

        # detection error bars: neutral grey, drawn underneath the markers
        ax.errorbar(dm[det], tau[det], yerr=[lo_plot[det], hi[det]],
                    xerr=(dm_err[det] if dm_err is not None else None),
                    fmt="none", ecolor=NO_FREQ_COLOR, elinewidth=1.0,
                    capsize=3, zorder=1)

        # detection markers: coloured if they have a frequency, else grey
        det_c  = det & np.isfinite(freq)
        det_nc = det & ~np.isfinite(freq)
        sc = ax.scatter(dm[det_c], tau[det_c], c=freq[det_c], cmap=cmap,
                        norm=norm, s=45, edgecolors="k", linewidths=0.3,
                        zorder=3)
        ax.scatter(dm[det_nc], tau[det_nc], c=NO_FREQ_COLOR, s=45,
                   edgecolors="k", linewidths=0.3, zorder=3)

        # upper limits: downward arrows, coloured by frequency (grey if none)
        for i in np.where(ul)[0]:
            col = cmap(norm(freq[i])) if np.isfinite(freq[i]) else NO_FREQ_COLOR
            ax.errorbar(dm[i], tau[i], yerr=ARROW_FRAC * tau[i], uplims=True,
                        xerr=(dm_err[i] if dm_err is not None else None),
                        fmt="none", ecolor=col, elinewidth=1.6, capsize=0,
                        zorder=2)

        # colourbar (built from an explicit mappable so it works even if there
        # happen to be no coloured detections)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Scattering frequency (GHz)")

        # legend: shapes only (colour is carried by the colourbar)
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=NO_FREQ_COLOR, markeredgecolor="k",
                   markersize=7, label="Detection"),
            Line2D([0], [0], marker="v", linestyle="none",
                   color=NO_FREQ_COLOR, markersize=8, label="Upper limit"),
        ]
        ax.legend(handles=handles, frameon=False)

    # ---- shared axis setup ----
    if LOG_Y:
        ax.set_yscale("log")
    if LOG_X:
        ax.set_xscale("log")
    ax.set_xlabel(r"Dispersion measure  DM  (pc cm$^{-3}$)")
    ax.set_ylabel(r"Scattering time  $\tau$  (ms)")
    ax.set_title(TITLE)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"Plotted {det.sum()} detections and {ul.sum()} upper limits "
          f"({(~good).sum()} rows skipped). Saved -> {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()