"""
Plot FRB scattering timescale (tau, ms) versus extragalactic dispersion measure
(DM_exgal, pc cm^-3).

  * Extragalactic DM on the x-axis, tau on the y-axis.
  * Detections are drawn as points with asymmetric vertical error bars
    (independent + and - errors).
  * Upper limits -- any tau value written in the CSV as "<value"
    (e.g. "< 0.15" or "<0.4") -- are drawn as downward arrows instead of
    points, with the tail of the arrow sitting at the limiting value.
  * Optionally, points and arrows are colour-coded by the frequency (GHz) at
    which the scattering was measured.
  * Optionally, tau values are converted to a reference frequency using
    tau_ref = tau_obs * (REF_FREQ_GHZ / freq_obs)^SCATTERING_INDEX.
  * Optionally, the Galactic pulsar tau-DM relation is overplotted as a
    reference band. See the "GALACTIC PULSAR RELATION" notes below.

Only the CONFIG block below should need editing. Set the column names to match
your catalogue, then run:  python plot_tau_vs_dm.py

Extra columns in the file are ignored -- only the columns named below are used.
Rows with a missing/unreadable DM or tau are skipped and counted.


GALACTIC PULSAR RELATION -- attribution and caveats
---------------------------------------------------
The curve drawn here is Equation (8) of Cordes, Ocker & Chatterjee (2022),
ApJ 931, 88 -- a fit of the Ramachandran et al. (1997) functional form to a
compilation of 568 Galactic pulsars (measurements + upper limits), with tau
scaled to 1 GHz assuming x_tau = 4:

    tau_psr(DM, nu) = 1.90e-7 ms * nu^-x_tau * DM^1.5 * (1 + 3.55e-5 * DM^3.0)

The +/-1 sigma band uses sigma_log(tau) = 0.76 dex, which IS from
Bhat et al. (2004), ApJ 605, 759 (bibcode 2004ApJ...605..759B).
So: cite CO&C22 for the CURVE, Bhat+2004 for the BAND WIDTH.

Two corrections/caveats are built in:

  1. PLANE-WAVE BOOST (x3). Pulsars sit inside the scattering medium and emit
     spherical wavefronts; FRBs arrive as plane waves. CO&C22 shift the pulsar
     band UP by a factor of 3 before comparing it to FRBs (their Fig. 5 vs.
     Fig. 4). PLANE_WAVE_BOOST = 3.0 does this. Set it to 1.0 to reproduce the
     raw pulsar fit instead.

  2. THE X-AXIS IS NOT APPLES-TO-APPLES. For pulsars, DM and tau come from the
     SAME turbulent disk, so a tau-DM relation is physically meaningful. For
     FRBs, tau is dominated by the host galaxy's ISM while DM_exgal is
     dominated by the weakly-scattering IGM. Strictly the abscissa should be
     DM_host in the host rest frame (CO&C22 say this explicitly, and then use
     total DM anyway because DM_host is unknown for most bursts).
     => The curve is a REFERENCE YARDSTICK, NOT A FIT OR A PREDICTION.
        FRBs are expected to fall well below it; the size of the offset is the
        measurement. Do not compute residuals against it.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================== CONFIG ===============================
CSV_PATH = "frb data and graphs (nexus 2026, chatterjee & cordes) - graph_tau_dm.csv"

# This sheet has a SINGLE header row (frb_name, tau_scatt, ...), so header=0
# keeps row 0 as the column names and we skip nothing. If you ever export a
# version with a second units row beneath the header, set SKIP_ROWS = [1].
HEADER_ROW = 0
SKIP_ROWS  = None

NA_VALUES  = [".."]                       # missing-value placeholder in the file

# --- Column names (exactly as they appear in the first header row) ---------
DM_COL     = "dm_exgal"                    # extragalactic dispersion measure, pc cm^-3
TAU_COL    = "tau_scatt"                   # scattering time, ms (may start "<")
TAU_ERR_HI = "tau_err_up"                  # +error on tau, ms  (blank/.. for limits)
TAU_ERR_LO = "tau_err_down"                # -error on tau, ms  (blank/.. for limits)
DM_ERR_COL = None                          # e.g. "dm_err" to draw x error bars, or None
FREQ_COL   = "scatt_freq"                  # frequency (GHz) tau was measured at
NAME_COL   = "frb_name"                    # used only to name bursts in warnings

# --- Frequency normalization ----------------------------------------------
# Convert every tau value and tau error bar to the reference frequency using
# tau ∝ frequency^SCATTERING_INDEX. CO&C22 adopt x_tau = 4 as fiducial, i.e.
# SCATTERING_INDEX = -4.0. Kolmogorov would be -4.4.
NORMALIZE_TO_REF_FREQ = True
REF_FREQ_GHZ = 1.0
SCATTERING_INDEX = -4.0

# --- Galactic pulsar tau-DM relation (CO&C 2022, Eq. 8) -------------------
#   tau(DM, nu) = PSR_A * nu^(-PSR_XTAU) * DM^PSR_ALPHA
#                 * (1 + PSR_B * DM^PSR_BETA)          [ms, nu in GHz]
SHOW_PSR_RELATION = True
PSR_A      = 1.90e-7      # ms
PSR_ALPHA  = 1.5          # DM index
PSR_B      = 3.55e-5      # turn-up coefficient
PSR_BETA   = 3.0          # turn-up index
PSR_XTAU   = 4.0          # frequency index OF THE RELATION (CO&C fiducial)

# Spherical-wave (pulsar, embedded) -> plane-wave (FRB) correction.
# CO&C22 shift the pulsar band up by 3 before comparing to FRBs.
# Set to 1.0 to draw the raw pulsar fit (their Fig. 4) instead.
PLANE_WAVE_BOOST = 3.0

PSR_SCATTER_DEX = 0.76    # sigma_log(tau), from Bhat et al. (2004)
SHOW_PSR_BAND   = True
PSR_COLOR       = "#0e7490"
PSR_STYLE       = "--"
PSR_WIDTH       = 1.6
PSR_BAND_ALPHA  = 0.15
# Frequency at which to draw the curve. Forced to REF_FREQ_GHZ when
# NORMALIZE_TO_REF_FREQ is True (the tau values are at that frequency).
PSR_CURVE_FREQ_GHZ = 1.0
# Extend the curve across the full x-axis rather than only the data span.
PSR_SPAN_FULL_AXIS = True

# --- Appearance ------------------------------------------------------------
LOG_Y      = True    # tau usually spans orders of magnitude -> log is clearer
LOG_X      = True    # log DM strongly recommended when the psr curve is shown
ARROW_FRAC = 0.35    # downward-arrow length as a fraction of the tau value
CMAP       = "viridis"   # colormap used when COLOR_BY_FREQ is True
FREQ_LOG   = False       # use a log colour scale for frequency
COLOR_BY_FREQ = False    # set True to color-code points/arrows by frequency

# On a LOG y-axis, a lower error bar with err_down >= tau reaches toward zero
# and shoots to the bottom of the plot. If this is set to a number N, the drawn
# lower error is capped so the bar extends at most N decades below the point
# (purely cosmetic -- the data is unchanged). Leave None to draw errors as-is.
CLIP_LOWER_ERR_DECADES = None

TITLE      = "FRB scattering time at 1 GHz vs. extragalactic dispersion measure"
OUT_PNG    = "tau_vs_dm.png"
DPI        = 200
NO_FREQ_COLOR = "0.6"    # grey used for points/arrows that have no frequency
POINT_SIZE = 32          # marker area for detections

# --- Trendline -------------------------------------------------------------
# Fits detections only; upper limits are excluded from the fit.
# NOTE: a naive fit that ignores the upper limits is biased HIGH. Treat any
# trendline here as illustrative, not as a censored-data regression.
SHOW_TRENDLINE = False
TRENDLINE_COLOR = "#64748b"
TRENDLINE_STYLE = "-"
TRENDLINE_WIDTH = 1.5
TRENDLINE_LABEL = None   # None -> label describes whether log axes were used
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


def psr_tau(dm, nu_ghz):
    """Galactic pulsar tau-DM relation, Cordes, Ocker & Chatterjee (2022) Eq. 8.

    dm     : dispersion measure, pc cm^-3 (array-like, > 0)
    nu_ghz : frequency in GHz
    returns: scattering time in ms, INCLUDING the spherical->plane wave boost
             so the curve is directly comparable to FRB scattering times.
    """
    dm = np.asarray(dm, dtype=float)
    tau = (PSR_A
           * nu_ghz ** (-PSR_XTAU)
           * dm ** PSR_ALPHA
           * (1.0 + PSR_B * dm ** PSR_BETA))
    return PLANE_WAVE_BOOST * tau


def add_psr_relation(ax, dm, good):
    """Overplot the Galactic pulsar relation. Returns a list of legend handles."""
    if not SHOW_PSR_RELATION:
        return []

    # Which frequency does the curve correspond to?
    if NORMALIZE_TO_REF_FREQ:
        nu = REF_FREQ_GHZ
    else:
        nu = PSR_CURVE_FREQ_GHZ
        print("NOTE: NORMALIZE_TO_REF_FREQ is False, so the plotted tau values "
              f"sit at their observed frequencies while the pulsar curve is "
              f"drawn at {nu:g} GHz. The comparison is only meaningful if every "
              "tau happens to be at that frequency; otherwise set "
              "NORMALIZE_TO_REF_FREQ = True.")

    # x-range for the curve
    if PSR_SPAN_FULL_AXIS:
        xlo, xhi = ax.get_xlim()
    else:
        dm_good = dm[good & np.isfinite(dm) & (dm > 0)]
        if dm_good.size == 0:
            print("WARNING: no usable DM values; pulsar curve skipped.")
            return []
        xlo, xhi = dm_good.min(), dm_good.max()

    xlo = max(xlo, 1.0)          # relation is only sensible for DM >~ 1
    if xhi <= xlo:
        print("WARNING: degenerate DM range; pulsar curve skipped.")
        return []

    dm_grid = np.logspace(np.log10(xlo), np.log10(xhi), 400)
    tau_psr = psr_tau(dm_grid, nu)

    handles = []

    if SHOW_PSR_BAND and PSR_SCATTER_DEX:
        ax.fill_between(dm_grid,
                        tau_psr * 10.0 ** (-PSR_SCATTER_DEX),
                        tau_psr * 10.0 ** (+PSR_SCATTER_DEX),
                        color=PSR_COLOR, alpha=PSR_BAND_ALPHA,
                        linewidth=0, zorder=0)
        handles.append(
            Patch(facecolor=PSR_COLOR, alpha=PSR_BAND_ALPHA, edgecolor="none",
                  label=rf"$\pm${PSR_SCATTER_DEX:g} dex (Bhat+ 2004)")
        )

    boost_note = "" if PLANE_WAVE_BOOST == 1.0 else ", plane-wave corr."
    (line,) = ax.plot(dm_grid, tau_psr, color=PSR_COLOR,
                      linestyle=PSR_STYLE, linewidth=PSR_WIDTH, zorder=1,
                      label=(rf"Galactic pulsars (Cordes+ 2022{boost_note})"))
    handles.insert(0, line)

    return handles


def add_trendline(ax, dm, tau, det):
    """Fit and draw a first-order trendline using the plotted axis scaling."""
    if not SHOW_TRENDLINE:
        return None

    fit = det & np.isfinite(dm) & np.isfinite(tau)
    if LOG_X:
        fit &= dm > 0
    if LOG_Y:
        fit &= tau > 0
    if fit.sum() < 2:
        print("WARNING: fewer than two detections available; trendline skipped.")
        return None

    x_fit = np.log10(dm[fit]) if LOG_X else dm[fit]
    y_fit = np.log10(tau[fit]) if LOG_Y else tau[fit]
    slope, intercept = np.polyfit(x_fit, y_fit, 1)

    if LOG_X:
        x_line = np.logspace(np.log10(dm[fit].min()), np.log10(dm[fit].max()), 200)
        x_eval = np.log10(x_line)
    else:
        x_line = np.linspace(dm[fit].min(), dm[fit].max(), 200)
        x_eval = x_line

    y_line = slope * x_eval + intercept
    if LOG_Y:
        y_line = 10.0 ** y_line

    if TRENDLINE_LABEL is not None:
        label = TRENDLINE_LABEL
    elif LOG_X and LOG_Y:
        label = "Trendline (detections, log-log fit)"
    elif LOG_Y:
        label = "Trendline (detections, log-y fit)"
    elif LOG_X:
        label = "Trendline (detections, log-x fit)"
    else:
        label = "Trendline (detections)"

    (line,) = ax.plot(x_line, y_line, color=TRENDLINE_COLOR,
                      linestyle=TRENDLINE_STYLE, linewidth=TRENDLINE_WIDTH,
                      label=label, zorder=4)
    return line


def normalize_tau_to_ref_freq(tau, hi, lo, freq, names):
    """Scale tau and tau error bars from observed frequency to REF_FREQ_GHZ."""
    if not NORMALIZE_TO_REF_FREQ:
        return tau, hi, lo
    if FREQ_COL is None:
        raise ValueError("NORMALIZE_TO_REF_FREQ=True requires FREQ_COL to be set.")

    valid_freq = np.isfinite(freq) & (freq > 0)
    if (~valid_freq & np.isfinite(tau)).any():
        print("WARNING: these rows have tau values but missing/non-positive "
              "frequencies, so they will be skipped during 1 GHz normalization:")
        for i in np.where(~valid_freq & np.isfinite(tau))[0]:
            print(f"  {names[i]}: tau={tau[i]:g} ms, frequency={freq[i]}")

    scale = np.full_like(tau, np.nan, dtype=float)
    scale[valid_freq] = (REF_FREQ_GHZ / freq[valid_freq]) ** SCATTERING_INDEX

    return tau * scale, hi * scale, lo * scale


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

    names = (df[NAME_COL].astype(str).to_numpy()
             if (NAME_COL is not None and NAME_COL in df.columns)
             else np.array([f"row {i}" for i in range(len(df))]))

    if NORMALIZE_TO_REF_FREQ:
        tau, hi, lo = normalize_tau_to_ref_freq(tau, hi, lo, freq, names)

    # ---- keep only rows with a usable DM and tau ----
    good = np.isfinite(tau) & np.isfinite(dm)
    if LOG_Y:
        good &= tau > 0
    if LOG_X:
        good &= dm > 0

    det = good & ~is_upper                     # detections
    ul  = good & is_upper                      # upper limits

    # ---- warn about lower error bars that would cross zero on a log axis ----
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

    if not COLOR_BY_FREQ:
        # -------- simple two-colour scheme (no frequency info) --------
        ax.errorbar(dm[det], tau[det], yerr=[lo_plot[det], hi[det]],
                    xerr=(dm_err[det] if dm_err is not None else None),
                    fmt="o", ms=np.sqrt(POINT_SIZE), color="#1f6feb",
                    ecolor="#1f6feb", capsize=3, elinewidth=1.1,
                    label="Detection", zorder=3)
        ax.errorbar(dm[ul], tau[ul],
                    yerr=ARROW_FRAC * tau[ul], uplims=True,
                    xerr=(dm_err[ul] if dm_err is not None else None),
                    fmt="none", ecolor="#d1495b", elinewidth=1.4,
                    capsize=0, label="Upper limit", zorder=2)
        trendline = add_trendline(ax, dm, tau, det)

        # Axis scales must be set BEFORE the pulsar curve so that, when
        # PSR_SPAN_FULL_AXIS is True, get_xlim() returns the final log limits.
        if LOG_Y:
            ax.set_yscale("log")
        if LOG_X:
            ax.set_xscale("log")
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        psr_handles = add_psr_relation(ax, dm, good)

        # The relation climbs steeply and would otherwise blow up the y-range,
        # so pin the limits back to what the DATA needs.
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        handles, _ = ax.get_legend_handles_labels()
        handles = [h for h in handles if h not in psr_handles]
        ax.legend(handles=handles + psr_handles, frameon=False, fontsize=9,
                  loc="best")

    else:
        # -------- colour-coded by scattering frequency --------
        if freq is None:
            raise ValueError("COLOR_BY_FREQ=True requires FREQ_COL to be set.")
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
                    capsize=3, zorder=2)

        # detection markers: coloured if they have a frequency, else grey
        det_c  = det & np.isfinite(freq)
        det_nc = det & ~np.isfinite(freq)
        ax.scatter(dm[det_c], tau[det_c], c=freq[det_c], cmap=cmap,
                   norm=norm, s=POINT_SIZE, edgecolors="k", linewidths=0.3,
                   zorder=3)
        ax.scatter(dm[det_nc], tau[det_nc], c=NO_FREQ_COLOR, s=POINT_SIZE,
                   edgecolors="k", linewidths=0.3, zorder=3)

        # upper limits: downward arrows, coloured by frequency (grey if none)
        for i in np.where(ul)[0]:
            col = cmap(norm(freq[i])) if np.isfinite(freq[i]) else NO_FREQ_COLOR
            ax.errorbar(dm[i], tau[i], yerr=ARROW_FRAC * tau[i], uplims=True,
                        xerr=(dm_err[i] if dm_err is not None else None),
                        fmt="none", ecolor=col, elinewidth=1.6, capsize=0,
                        zorder=2)

        # colourbar (explicit mappable, so it works even with no coloured points)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Scattering frequency (GHz)")

        trendline = add_trendline(ax, dm, tau, det)

        if LOG_Y:
            ax.set_yscale("log")
        if LOG_X:
            ax.set_xscale("log")
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        psr_handles = add_psr_relation(ax, dm, good)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # legend: shapes only (colour is carried by the colourbar)
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=NO_FREQ_COLOR, markeredgecolor="k",
                   markersize=np.sqrt(POINT_SIZE), label="Detection"),
            Line2D([0], [0], marker="v", linestyle="none",
                   color=NO_FREQ_COLOR, markersize=8, label="Upper limit"),
        ]
        if trendline is not None:
            handles.append(trendline)
        handles += psr_handles
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")

    # ---- shared axis setup ----
    ax.set_xlabel(r"Extragalactic dispersion measure  DM$_{\rm exgal}$  (pc cm$^{-3}$)")
    if NORMALIZE_TO_REF_FREQ:
        ax.set_ylabel(
            rf"Scattering time  $\tau_{{{REF_FREQ_GHZ:g}\,\rm GHz}}$  (ms)"
        )
    else:
        ax.set_ylabel(r"Scattering time  $\tau$  (ms)")
    ax.set_title(TITLE)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"Plotted {det.sum()} detections and {ul.sum()} upper limits "
          f"({(~good).sum()} rows skipped). Saved -> {OUT_PNG}")
    if SHOW_PSR_RELATION:
        print("Pulsar curve: Cordes, Ocker & Chatterjee (2022) Eq. 8"
              f"{'' if PLANE_WAVE_BOOST == 1.0 else f' x{PLANE_WAVE_BOOST:g} plane-wave'}"
              f"; band = +/-{PSR_SCATTER_DEX:g} dex (Bhat et al. 2004).")
        print("It is a Galactic reference yardstick drawn against DM_exgal -- "
              "NOT a fit or a prediction for FRBs.")
    plt.show()


if __name__ == "__main__":
    main()