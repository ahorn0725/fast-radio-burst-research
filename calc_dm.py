#!/usr/bin/env python3
"""
frb_dm_exgal_ne2025.py
----------------------
For each burst in the (filled) FRB CSV, use the NE2025 Galactic electron-density
model (Cordes & Ocker, via the `mwprop` package) with the burst's (l, b) to
compute the Milky Way DISK dispersion measure along that line of sight, then
form the extragalactic DM:

        DM_exgal = DM_obs - DM_disk            (default, halo kept separate)
        DM_exgal = DM_obs - DM_disk - DM_halo  (if SUBTRACT_HALO = True)

Adds three columns and leaves every original column untouched:
        "DM ne2025 (pc cm^-3)"   -> NE2025 Milky Way disk DM
        "DM MW halo (pc cm^-3)"  -> fixed halo value (stored, not necessarily subtracted)
        "DM exgal (pc cm^-3)"    -> DM_obs minus the above per SUBTRACT_HALO

Run
---
    python frb_dm_exgal_ne2025.py dm_ex_vs_z_filled.csv
(writes <input>_ne2025.csv if no output name is given)

Needs: numpy, pandas, mwprop   (pip install mwprop)

Note: the pure-Python NE2025 is ~45x slower than the Fortran build, but for ~100
bursts that's still only a minute or two. A short progress line prints as it goes.
"""

import argparse
import warnings

# mwprop loads its model parameter files on import. Do the star import the way
# the package docs show (it sets up config), THEN bind numpy/pandas so their
# names aren't shadowed.
try:
    from mwprop.nemod import *                      # noqa: F401,F403  (loads NE2025 config)
    from mwprop.nemod.NE2025 import ne2025
except ImportError as e:
    raise SystemExit(
        "Could not import mwprop's NE2025. Install it first:\n"
        "    pip install mwprop\n"
        f"(original error: {e})"
    )

import numpy as np
import pandas as pd


# ------------------------------- CONFIG ------------------------------------- #

LOS_DIST_KPC  = 30.0      # integrate the sightline this far (kpc). Past ~25-30 kpc
                          # the disk model adds ~0, so DM saturates at the max
                          # Galactic value -- which is what you want for an FRB.

HALO_DM       = 50.0      # fixed Milky Way halo DM (pc cm^-3), stored as its own
                          # column so you can subtract it later if you choose.

SUBTRACT_HALO = False     # False -> DM_exgal = DM_obs - DM_disk   (halo kept aside)
                          # True  -> DM_exgal = DM_obs - DM_disk - HALO_DM

ROUND_DM      = 2

COL_DM_DISK  = "DM ne2025 (pc cm^-3)"
COL_DM_HALO  = "DM MW halo (pc cm^-3)"
COL_DM_EXGAL = "DM exgal (pc cm^-3)"

MISSING = {"", "..", "...", "nan", "n/a", "na", "-"}


# ------------------------------- helpers ------------------------------------ #

def is_missing(v):
    return v is None or str(v).strip().lower() in MISSING


def as_float(v):
    if is_missing(v):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def find_col(cols, *cands):
    """Return the actual column label matching any candidate (case-insensitive)."""
    low = {str(c).strip().lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _dm_from_output(Dv):
    """Pull the DM value out of NE2025's value dictionary, robust to key naming."""
    if "DM" in Dv:
        return float(Dv["DM"])
    for k in Dv:
        if "dm" in str(k).lower():
            return float(Dv[k])
    raise KeyError(f"No DM key found in NE2025 output. Available keys: {list(Dv)}")


def mw_dm_ne2025(l, b, dist_kpc=LOS_DIST_KPC):
    """Milky Way disk DM (pc cm^-3) toward Galactic (l, b) integrated to dist_kpc.
    ndir < 0 tells NE2025 to go distance -> DM (dmd is then a distance in kpc)."""
    Dk, Dv, Du, Dd = ne2025(ldeg=float(l), bdeg=float(b),
                            dmd=float(dist_kpc), ndir=-1,
                            classic=False, dmd_only=True)
    return _dm_from_output(Dv)


# --------------------------------- main ------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Add NE2025 Milky Way DM and extragalactic DM columns to an FRB CSV.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    args = ap.parse_args()
    out_path = args.output or args.input.rsplit(".", 1)[0] + "_ne2025.csv"

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    cols = list(df.columns)

    c_l   = find_col(cols, "l (deg)", "l(deg)", "l_deg", "l")
    c_b   = find_col(cols, "b (deg)", "b(deg)", "b_deg", "b")
    c_dm  = find_col(cols, "DM obs (pc cm^-3)", "DM obs", "dm_obs")
    c_frb = find_col(cols, "FRB", "frb", "name")
    if not (c_l and c_b and c_dm):
        raise SystemExit(
            f"Missing a required column. Found l={c_l!r}, b={c_b!r}, DM_obs={c_dm!r}. "
            "Check the header names against COLUMN candidates in find_col().")

    for c in (COL_DM_DISK, COL_DM_HALO, COL_DM_EXGAL):
        if c not in df.columns:
            df[c] = ""

    # pandas 3 uses strict string columns when dtype=str, including the result
    # columns created above. Calculated DM values are floats, so convert every
    # column after setup to allow both strings and numbers.
    df = df.astype(object)

    warnings.filterwarnings("ignore")  # quiet NE2025's per-clump/void notices for the batch

    n_done = 0
    n_skipped = 0
    negatives = []          # bursts where DM_exgal < 0 (model over-predicts the MW)
    for i in range(len(df)):
        l = as_float(df.at[i, c_l])
        b = as_float(df.at[i, c_b])
        if l is None or b is None:
            n_skipped += 1
            continue        # no coordinates -> leave the new cells blank

        dm_disk = mw_dm_ne2025(l, b)
        df.at[i, COL_DM_DISK] = round(dm_disk, ROUND_DM)
        df.at[i, COL_DM_HALO] = HALO_DM

        dmobs = as_float(df.at[i, c_dm])
        if dmobs is not None:
            subtract = dm_disk + (HALO_DM if SUBTRACT_HALO else 0.0)
            dm_exgal = dmobs - subtract
            df.at[i, COL_DM_EXGAL] = round(dm_exgal, ROUND_DM)
            if dm_exgal < 0:
                name = df.at[i, c_frb] if c_frb else f"row {i + 2}"
                negatives.append(name)

        n_done += 1
        if n_done % 25 == 0:
            print(f"  ...{n_done} bursts done")

    df.to_csv(out_path, index=False)

    mode = "DM_obs - DM_disk" + (" - DM_halo" if SUBTRACT_HALO else "")
    print(f"\nWrote {out_path}")
    print(f"  processed: {n_done} bursts   (skipped {n_skipped} with no l/b)")
    print(f"  DM_exgal = {mode}")
    print(f"  halo = {HALO_DM} pc/cm^3 "
          f"({'subtracted' if SUBTRACT_HALO else 'stored separately, NOT subtracted'})")
    if negatives:
        print("  NOTE - DM_exgal came out negative for these (NE2025 over-predicts "
              "the MW on these low-latitude sightlines; worth a look):")
        print("        " + ", ".join(str(x) for x in negatives))


if __name__ == "__main__":
    main()
