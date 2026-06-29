#!/usr/bin/env python3
"""
frb_coord_fill.py
-----------------
Fill in the ".." placeholder cells in an FRB catalogue CSV by converting
between coordinate frames. This script ONLY touches coordinates + their
errors:

    l (deg), l_err, b (deg), b_err
    ra (hms), ra (deg), ra_err_sec, ra_err_arcsec
    dec (dms), dec (deg), dec_err (")

Everything else (DM, z, etc.) is passed through untouched.

What it does per row
--------------------
1. Resolves a single definitive (ra_deg, dec_deg) and (l, b):
     - prefers measured equatorial coords if present (from ra/dec deg OR hms/dms)
     - otherwise converts FROM galactic l/b
2. Fills any blank coordinate cell from that solution:
     - ra(hms) <-> ra(deg),  dec(dms) <-> dec(deg)   (pure formatting)
     - (ra,dec) <-> (l,b)                            (ICRS <-> Galactic via astropy)
3. Fills error cells:
     - ra_err_sec <-> ra_err_arcsec  via  arcsec_onsky = sec * 15 * cos(dec)
     - (ra/dec errors) <-> (l/b errors) by rotating the on-sky covariance
       matrix through a numerically computed Jacobian (a pure rotation in the
       tangent plane).

Only "..", blank, or NaN cells are filled. Measured values are never overwritten.

Assumptions (flip in CONFIG if your catalogue differs)
------------------------------------------------------
- ra_err_arcsec is the ON-SKY angular error (includes the cos(dec) factor),
  i.e. ra_err_arcsec = ra_err_sec * 15 * cos(dec).
- l_err and b_err are coordinate-frame errors in DEGREES (std dev of the l and
  b values themselves). b_err is already on-sky; l_err is divided/multiplied by
  cos(b) internally to move between coordinate and on-sky angle.
- dec_err is in arcsec.
- Independent (diagonal) errors in the source frame; the rotation mixes them.

Run
---
    python frb_coord_fill.py input.csv output.csv
(if output is omitted it writes <input>_filled.csv)

Needs: numpy, pandas, astropy
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u


# ----------------------------- CONFIG --------------------------------------- #

MISSING_TOKENS = {"", "..", "...", "nan", "n/a", "na", "-"}

# How a blank dec should be written when we generate it: "symbol" -> 4°33'44.8",
# "letters" -> +04d33m44.80s  (letters parses back cleanly; symbols match your sheet)
DEC_STYLE = "symbol"

DECIMALS = {           # rounding for generated numbers
    "deg": 6,          # l, b, ra(deg), dec(deg)
    "l_err": 6,
    "b_err": 6,
    "dec_err": 3,      # arcsec
    "ra_err_sec": 4,
    "ra_err_arcsec": 3,
}

# Canonical key  ->  acceptable (normalized) header names in your CSV.
COLUMN_ALIASES = {
    "frb":           ["frb"],
    "l":             ["l (deg)", "l(deg)", "l_deg", "l"],
    "l_err":         ["l_err", "l err"],
    "b":             ["b (deg)", "b(deg)", "b_deg", "b"],
    "b_err":         ["b_err", "b err"],
    "ra_hms":        ["ra (hms)", "ra(hms)", "ra_hms"],
    "ra_deg":        ["ra (deg)", "ra(deg)", "ra_deg"],
    "ra_err_sec":    ["ra_err_sec", "ra err sec"],
    "ra_err_arcsec": ["ra_err_arcsec", "ra err arcsec"],
    "dec_dms":       ["dec (dms)", "dec(dms)", "dec_dms"],
    "dec_deg":       ["dec (deg)", "dec(deg)", "dec_deg"],
    "dec_err":       ['dec_err (")', "dec_err ('')", "dec_err", "dec err"],
}


# --------------------------- small helpers ---------------------------------- #

def norm(s):
    """Normalize a header/cell string for matching."""
    return " ".join(str(s).strip().lower().split())


def is_missing(val):
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return norm(val) in MISSING_TOKENS


def as_float(val):
    """Return float if cell holds a plain number, else None."""
    if is_missing(val):
        return None
    try:
        return float(str(val).strip())
    except ValueError:
        return None


def wrap180(x):
    """Wrap an angle difference (deg) into (-180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def parse_hms(s):
    """'21h44m25.255s' -> degrees. Pulls the h/m/s numbers out directly, so it
    doesn't matter which separator characters (h m s, colons, odd unicode) sit
    between them. A bare decimal in the hms column is read as decimal HOURS."""
    if is_missing(s):
        return None
    txt = str(s).strip().replace("\u2212", "-")  # normalize unicode minus
    if re.fullmatch(r"[+-]?\d+\.?\d*", txt):
        return float(txt) * 15.0
    nums = re.findall(r"\d+\.?\d*", txt)
    if not nums:
        return None
    h = float(nums[0])
    m = float(nums[1]) if len(nums) > 1 else 0.0
    sec = float(nums[2]) if len(nums) > 2 else 0.0
    return (h + m / 60.0 + sec / 3600.0) * 15.0


def parse_dms(s):
    """'4°33'44.8\"', '73d45'05\"', '-40°54'00.1″' -> degrees. Regex-based, so
    ANY flavour of minute/second mark ( '  '  d  m  s  prime  double-prime ...)
    parses fine. A bare decimal is taken as degrees."""
    if is_missing(s):
        return None
    txt = str(s).strip().replace("\u2212", "-")  # normalize unicode minus
    if re.fullmatch(r"[+-]?\d+\.?\d*", txt):
        return float(txt)
    sign = -1.0 if txt.lstrip().startswith("-") else 1.0
    nums = re.findall(r"\d+\.?\d*", txt)
    if not nums:
        return None
    d = float(nums[0])
    m = float(nums[1]) if len(nums) > 1 else 0.0
    sec = float(nums[2]) if len(nums) > 2 else 0.0
    return sign * (d + m / 60.0 + sec / 3600.0)


def fmt_hms(ra_deg):
    return Angle(ra_deg, u.deg).to_string(unit=u.hour, sep=("h", "m", "s"),
                                          precision=4, pad=True)


def fmt_dms(dec_deg):
    s = Angle(dec_deg, u.deg).to_string(unit=u.deg, sep=("d", "m", "s"),
                                        precision=2, alwayssign=True, pad=True)
    if DEC_STYLE == "symbol":
        s = s.replace("d", "°").replace("m", "'").replace("s", '"')
    return s


def r(val, key):
    return None if val is None else round(float(val), DECIMALS[key])


# --------------------- coordinate frame conversions ------------------------- #

def resolve_coords(get):
    """
    Resolve a single definitive (ra_deg, dec_deg, l, b) from whatever is present.
    `get(key)` returns the raw cell for a canonical key.

    Returns a dict with ra/dec/l/b on success, or {"_unresolved": True, "have":[...]}
    listing which pieces WERE found, so the caller can explain the gap.
    """
    ra = as_float(get("ra_deg"))
    if ra is None:
        ra = parse_hms(get("ra_hms"))
    dec = as_float(get("dec_deg"))
    if dec is None:
        dec = parse_dms(get("dec_dms"))

    l = as_float(get("l"))
    b = as_float(get("b"))

    # Preferred: measured equatorial -> galactic
    if ra is not None and dec is not None:
        try:
            gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
            return {"ra": ra, "dec": dec,
                    "l": l if l is not None else gal.l.deg,
                    "b": b if b is not None else gal.b.deg,
                    "source": "equatorial"}
        except Exception:
            pass  # bad value; fall through and try galactic instead

    # Fallback: galactic -> equatorial (keep any measured ra/dec we did have)
    if l is not None and b is not None:
        try:
            icrs = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
            return {"ra": ra if ra is not None else icrs.ra.deg,
                    "dec": dec if dec is not None else icrs.dec.deg,
                    "l": l, "b": b, "source": "galactic"}
        except Exception:
            pass

    have = [n for n, v in (("ra", ra), ("dec", dec), ("l", l), ("b", b))
            if v is not None]
    return {"_unresolved": True, "have": have}


def eq_to_gal_rotation(ra_deg, dec_deg, eps_arcsec=1.0):
    """
    Numerically computed Jacobian / rotation M such that
        [dl*cos(b); db] = M [d_alpha*cos(dec); d_dec]
    where all quantities are ON-SKY (true) angles. M is ~orthogonal (a rotation),
    so it correctly mixes the variances when we do M C M^T.
    """
    eps = eps_arcsec / 3600.0  # deg
    base = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs").galactic
    b0, l0 = base.b.deg, base.l.deg
    cosd = np.cos(np.radians(dec_deg))
    cosb = np.cos(np.radians(b0))

    east = SkyCoord((ra_deg + eps / cosd) * u.deg, dec_deg * u.deg,
                    frame="icrs").galactic
    north = SkyCoord(ra_deg * u.deg, (dec_deg + eps) * u.deg,
                     frame="icrs").galactic

    def col(g):
        return [wrap180(g.l.deg - l0) * cosb / eps, (g.b.deg - b0) / eps]

    ce, cn = col(east), col(north)
    return np.array([[ce[0], cn[0]],
                     [ce[1], cn[1]]])


# ----------------------------- error filling -------------------------------- #

def fill_row(row, cols, warnings, idx):
    """Mutate `row` (a dict of canonical_key -> value) in place."""
    def get(k):
        return row.get(k) if k in cols else None

    sol = resolve_coords(get)
    if sol.get("_unresolved"):
        have = sol["have"]
        name = row.get("frb", f"row {idx}")
        if have:
            warnings.append(f"{name}: only have [{', '.join(have)}] -- "
                            f"need a complete ra+dec OR l+b pair to derive the rest")
        else:
            warnings.append(f"{name}: no usable coordinates in this row")
        return

    ra, dec, l, b = sol["ra"], sol["dec"], sol["l"], sol["b"]

    # --- positions ---------------------------------------------------------- #
    fills = {
        "ra_hms":  (lambda: fmt_hms(ra)),
        "ra_deg":  (lambda: r(ra, "deg")),
        "dec_dms": (lambda: fmt_dms(dec)),
        "dec_deg": (lambda: r(dec, "deg")),
        "l":       (lambda: r(l, "deg")),
        "b":       (lambda: r(b, "deg")),
    }
    for key, maker in fills.items():
        if key in cols and is_missing(row.get(key)):
            row[key] = maker()

    # --- errors ------------------------------------------------------------- #
    cosd = np.cos(np.radians(dec))
    cosb = np.cos(np.radians(b))

    sec = as_float(get("ra_err_sec"))
    arc = as_float(get("ra_err_arcsec"))     # on-sky arcsec
    decerr = as_float(get("dec_err"))        # arcsec
    lerr = as_float(get("l_err"))            # deg (coordinate)
    berr = as_float(get("b_err"))            # deg (on-sky)

    # ra time-seconds <-> on-sky arcsec (needs dec only)
    if sec is not None and arc is None:
        arc = sec * 15.0 * cosd
    elif arc is not None and sec is None and abs(cosd) > 1e-9:
        sec = arc / (15.0 * cosd)

    have_eq = (arc is not None) and (decerr is not None)
    have_gal = (lerr is not None) and (berr is not None)

    need_gal = ("l_err" in cols and is_missing(row.get("l_err"))) or \
               ("b_err" in cols and is_missing(row.get("b_err")))
    need_eq = ("ra_err_sec" in cols and is_missing(row.get("ra_err_sec"))) or \
              ("ra_err_arcsec" in cols and is_missing(row.get("ra_err_arcsec"))) or \
              ("dec_err" in cols and is_missing(row.get("dec_err")))

    if need_gal and have_eq and not have_gal:
        M = eq_to_gal_rotation(ra, dec)
        C_eq = np.diag([arc ** 2, decerr ** 2])           # arcsec^2, on-sky
        C_gal = M @ C_eq @ M.T
        sig_l_onsky = np.sqrt(C_gal[0, 0])                # arcsec
        sig_b = np.sqrt(C_gal[1, 1])                      # arcsec
        if abs(cosb) > 1e-9:
            lerr = (sig_l_onsky / cosb) / 3600.0          # deg, coordinate
        berr = sig_b / 3600.0                             # deg

    elif need_eq and have_gal and not have_eq:
        M = eq_to_gal_rotation(ra, dec)
        sig_l_onsky = lerr * cosb * 3600.0                # arcsec, on-sky
        sig_b = berr * 3600.0                             # arcsec
        C_gal = np.diag([sig_l_onsky ** 2, sig_b ** 2])
        Minv = np.linalg.inv(M)
        C_eq = Minv @ C_gal @ Minv.T
        arc = np.sqrt(C_eq[0, 0])                         # arcsec, on-sky
        decerr = np.sqrt(C_eq[1, 1])                      # arcsec
        if abs(cosd) > 1e-9:
            sec = arc / (15.0 * cosd)

    elif (need_eq or need_gal) and not (have_eq or have_gal):
        name = row.get("frb", f"row {idx}")
        warnings.append(f"{name}: positions filled, but no complete error pair "
                        f"to propagate from (errors left blank)")

    # write filled error cells (only where currently missing)
    writes = {
        "ra_err_sec":    r(sec, "ra_err_sec"),
        "ra_err_arcsec": r(arc, "ra_err_arcsec"),
        "dec_err":       r(decerr, "dec_err"),
        "l_err":         r(lerr, "l_err"),
        "b_err":         r(berr, "b_err"),
    }
    for key, val in writes.items():
        if key in cols and is_missing(row.get(key)) and val is not None:
            row[key] = val


# ------------------------------- IO / main ---------------------------------- #

def detect_header_row(raw):
    """Find the row index whose cells include 'frb' (handles a colored
    group-label row sitting above the real header)."""
    for i in range(min(5, len(raw))):
        if any(norm(c) == "frb" for c in raw.iloc[i].tolist()):
            return i
    return 0


def map_columns(header):
    """canonical key -> actual column label in the file."""
    normed = {norm(h): h for h in header}
    mapping = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in normed:
                mapping[key] = normed[a]
                break
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Fill '..' coordinate cells in an FRB CSV.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    args = ap.parse_args()
    out_path = args.output or args.input.rsplit(".", 1)[0] + "_filled.csv"

    raw = pd.read_csv(args.input, header=None, dtype=str, keep_default_na=False)
    hdr = detect_header_row(raw)
    header = raw.iloc[hdr].tolist()
    # pandas 3 may infer a strict StringDtype here. Generated coordinate cells
    # include floats, so use object columns to allow both strings and numbers.
    df = raw.iloc[hdr + 1:].copy().astype(object)
    df.columns = header
    df.reset_index(drop=True, inplace=True)

    colmap = map_columns(header)            # canonical -> actual label
    inv = {v: k for k, v in colmap.items()} # actual label -> canonical
    found = set(colmap)
    print("Matched columns:", {k: colmap[k] for k in sorted(found)})
    missing_cols = set(COLUMN_ALIASES) - found
    if missing_cols:
        print("Not found (left untouched):", sorted(missing_cols))

    warnings = []
    lb_missing = []        # FRBs that still lack l and/or b after processing
    for i in range(len(df)):
        # build canonical-keyed view of the row
        row = {}
        for label in df.columns:
            if label in inv:
                row[inv[label]] = df.iat[i, df.columns.get_loc(label)]
        fill_row(row, found, warnings, i + hdr + 2)  # +2 -> 1-based sheet row
        # write back
        for key, val in row.items():
            df.iat[i, df.columns.get_loc(colmap[key])] = "" if val is None else val
        # track l/b completeness (only if those columns exist at all)
        if "l" in found and "b" in found:
            if is_missing(row.get("l")) or is_missing(row.get("b")):
                lb_missing.append(row.get("frb") or f"row {i + hdr + 2}")

    df.to_csv(out_path, index=False)
    n = len(df)
    print(f"\nWrote {out_path}  ({n} rows)")
    if "l" in found and "b" in found:
        print(f"l/b present for {n - len(lb_missing)}/{n} rows.")
        if lb_missing:
            print("Still missing l/b ->", ", ".join(str(x) for x in lb_missing))
    if warnings:
        print("\nNotes:")
        for w in warnings:
            print("  -", w)


if __name__ == "__main__":
    main()