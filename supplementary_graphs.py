#!/usr/bin/env python3
"""
plot_dm_vs_l.py -- scatter Galactic latitude against dispersion measure.

Everything is configured in the SETTINGS block below. Set CSV_PATH, then just
run the file (Run button in VSCode, or `python plot_dm_vs_l.py`). No arguments
needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================== SETTINGS ==============================
# Path to your CSV. Either an absolute path, or a name relative to this
# script's folder (both are tried, so it works wherever VSCode runs from).
CSV_PATH = "DM_Z_GRAPHING_FRBS (1).csv"

# Column names in that CSV.
L_COL = "b_deg"          # Galactic coordinate, degrees -> x axis
DM_COL = "dm_obs"        # dispersion measure, pc/cm^3 -> y axis

# Optional extras. Set to None to skip.
COLOR_COL = None         # e.g. "b_deg" -- colours points, adds a colourbar
LABEL_COL = None         # e.g. "frb_name" -- annotates each point

# Display options.
WRAP_LONGITUDE = True    # True -> remap coordinates to [-180, 180), centred on 0
LOG_DM_AXIS = False      # True -> logarithmic y axis
INVERT_X = False         # True -> longitude increases leftwards, as on a sky map
TITLE = None             # None -> auto-generated title

# Output. OUTFILE = None opens an interactive window instead of saving.
OUTFILE = None           # e.g. "dm_vs_l.png"
DPI = 150
# ======================================================================


def resolve_csv(path_str):
    """Find the CSV as given, or relative to this script's folder."""
    candidate = Path(path_str).expanduser()
    if candidate.is_file():
        return candidate
    beside_script = Path(__file__).resolve().parent / path_str
    if beside_script.is_file():
        return beside_script
    sys.exit(
        f"error: could not find CSV '{path_str}'\n"
        f"  looked in: {candidate.resolve()}\n"
        f"         and: {beside_script}"
    )


def load_columns(path, wanted):
    """Read the CSV and check that every requested column is present."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        sys.exit(f"error: could not read {path}: {exc}")

    missing = [c for c in wanted if c is not None and c not in df.columns]
    if missing:
        sys.exit(
            "error: column(s) not found: " + ", ".join(missing)
            + "\navailable columns: " + ", ".join(map(str, df.columns))
        )
    return df


def main():
    csv_path = resolve_csv(CSV_PATH)
    df = load_columns(csv_path, [L_COL, DM_COL, COLOR_COL, LABEL_COL])
    n_in = len(df)

    # Coerce the numeric columns; anything unparseable becomes NaN.
    df[L_COL] = pd.to_numeric(df[L_COL], errors="coerce")
    df[DM_COL] = pd.to_numeric(df[DM_COL], errors="coerce")
    required = [L_COL, DM_COL]
    if COLOR_COL:
        df[COLOR_COL] = pd.to_numeric(df[COLOR_COL], errors="coerce")
        required.append(COLOR_COL)

    # Drop incomplete rows as a unit, so x, y, colour and label stay aligned.
    df = df.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_in - len(df)
    if n_dropped:
        print(f"dropped {n_dropped} of {n_in} rows with missing/non-numeric values")
    if df.empty:
        sys.exit("error: no plottable rows remain")

    lon = df[L_COL]
    if WRAP_LONGITUDE:
        lon = ((lon + 180.0) % 360.0) - 180.0
        xlim, ticks = (-180, 180), range(-180, 181, 60)
    else:
        lon = lon % 360.0
        xlim, ticks = (0, 360), range(0, 361, 60)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if COLOR_COL:
        sc = ax.scatter(lon, df[DM_COL], c=df[COLOR_COL], cmap="viridis",
                        s=38, edgecolor="k", linewidth=0.4, zorder=3)
        fig.colorbar(sc, ax=ax, label=COLOR_COL)
    else:
        ax.scatter(lon, df[DM_COL], s=38, color="tab:blue",
                   edgecolor="k", linewidth=0.4, zorder=3)

    if LABEL_COL:
        # Iterate over the cleaned frame so every label matches its own row.
        for x, y, text in zip(lon, df[DM_COL], df[LABEL_COL]):
            ax.annotate(str(text), (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_xticks(list(ticks))
    if INVERT_X:
        ax.invert_xaxis()
    if LOG_DM_AXIS:
        ax.set_yscale("log")

    ax.set_xlabel(r"Galactic latitude  $b$  (deg)")
    ax.set_ylabel(r"Dispersion measure  DM  (pc cm$^{-3}$)")
    ax.set_title(TITLE or f"DM vs Galactic latitude  ({len(df)} sources)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if OUTFILE:
        fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
        print(f"wrote {OUTFILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
