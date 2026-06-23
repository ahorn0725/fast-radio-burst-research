#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frb_catalogue.py
================
Ingest a CSV of FRB properties, fill in whatever can be derived, and write a new
CSV (originals untouched). Designed to run locally in VSCode.

What it does, per burst:
  1. Parse messy sexagesimal RA/Dec (handles  ° d ' ′ '' " ″  and stray spaces)
     -> decimal degrees. Fill l/b from RA/Dec, or RA/Dec from l/b, whichever is
     missing. Never overwrites a value you already provided.
  2. Look up the Milky Way (disk) DM toward the burst with the NE2025 model
     (mwprop). NE2025 needs Galactic l/b, which step 1 guarantees if any
     position exists.
  3. dm_exgal = dm_obs - dm_milky_way   (halo NOT subtracted; recorded separately).
  4. Propagate errors. Equatorial position errors (ra_err_sec / ra_err_arcsec /
     dec_err, OR an a_err/b_err/angle ellipse) are converted to on-sky l/b errors
     by rotating the covariance through the ICRS->Galactic frame.
  5. Tag provenance: every derived column has a sibling *_origin column so you can
     filter the spreadsheet to see exactly which values were measured vs computed.

Outputs are appended as new "*_calc" / "*_origin" columns; your 31 original
columns are kept verbatim.

----------------------------------------------------------------------
SETUP (run once in your terminal / VSCode integrated terminal):
    pip install mwprop pandas astropy numpy
USAGE:
    - Set INPUT_CSV below to your file path, then run:  python frb_catalogue.py
    - Or import and call process_csv("yourfile.csv") from another script.
    - For a single unpublished burst, see manual_burst() at the bottom.
----------------------------------------------------------------------
"""

import os
import re
import math
import warnings
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG  -- edit these
# ----------------------------------------------------------------------
INPUT_CSV   = "frb_data.csv"     # <-- your input file
OUTPUT_CSV  = None               # None -> "<input>_processed.csv"

MW_FRAC_ERR = 0.20               # assumed fractional uncertainty on the NE2025 disk DM
DM_HALO     = 50.0               # pc/cm^3, recorded only (NOT subtracted from dm_exgal)
NE2025_DIST_KPC = 100.0          # integrate electron density out to here (beyond the Galaxy)
NE2025_DM_KEY   = "DM"           # dict key NE2025 returns the integrated DM under.
                                 # If you get a KeyError, run `test_ne2025p.py` (ships with
                                 # mwprop) and check the printed keys, then set this.
ELLIPSE_NSIGMA  = 1.0            # a_err/b_err are quoted as this many sigma (1 = 1-sigma semi-axes)

# Strings that mean "empty" in your sheet
MISSING = {"", "n/a", "na", "..", "...", ".", "?", "nan", "none", "-", "—", "–"}

# ----------------------------------------------------------------------
# Optional dependencies, imported lazily with clear messages
# ----------------------------------------------------------------------
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _HAVE_ASTROPY = True
except Exception:
    _HAVE_ASTROPY = False

try:
    from mwprop.nemod.NE2025 import ne2025
    _HAVE_MWPROP = True
except Exception:
    _HAVE_MWPROP = False


# ======================================================================
# Small parsing helpers (validated against the sample data)
# ======================================================================
def clean(s):
    """Return a stripped string, or None if it is a missing-value marker."""
    if s is None:
        return None
    if isinstance(s, float) and math.isnan(s):
        return None
    t = str(s).strip()
    return None if t.lower() in MISSING else t


def to_float(s):
    t = clean(s)
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_ra_hms(s):
    """'22h15m37.712s' (and spaced/partial variants) -> RA in decimal degrees."""
    t = clean(s)
    if t is None:
        return None
    t = t.replace(" ", "")
    m = re.match(r'^([+-]?\d+(?:\.\d+)?)h(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?$', t, re.I)
    if not m:
        try:
            return float(t)  # maybe already decimal degrees
        except ValueError:
            return None
    h, mi, se = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return 15.0 * (h + mi / 60.0 + se / 3600.0)


def parse_dec_dms(s):
    """Parse Dec with any of  ° d  /  ' ′  /  '' \" ″  separators -> decimal degrees."""
    t = clean(s)
    if t is None:
        return None
    # normalise unicode + ascii separators to d / m / s
    t = (t.replace("′", "'").replace("’", "'").replace("″", '"').replace("“", '"'))
    t = t.replace("°", "d").replace("º", "d")
    t = t.replace("''", '"').replace("'", "m").replace('"', "s")
    t = t.replace(" ", "")
    m = re.match(r'^([+-]?)(\d+(?:\.\d+)?)d(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?$', t, re.I)
    if not m:
        try:
            return float(clean(s))
        except (ValueError, TypeError):
            return None
    sign = -1.0 if m.group(1) == '-' else 1.0
    d, mi, se = float(m.group(2)), float(m.group(3)), float(m.group(4))
    return sign * (d + mi / 60.0 + se / 3600.0)


# ======================================================================
# Coordinate transforms (astropy)
# ======================================================================
def radec_to_lb(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g = c.galactic
    return float(g.l.deg), float(g.b.deg)


def lb_to_radec(l_deg, b_deg):
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    i = c.icrs
    return float(i.ra.deg), float(i.dec.deg)


def fmt_hmsdms(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ra_str = c.ra.to_string(unit=u.hour, sep="hms", precision=3, pad=True)
    dec_str = c.dec.to_string(unit=u.deg, sep="dms", precision=2, alwayssign=True, pad=True)
    return ra_str, dec_str


# ======================================================================
# Positional error handling
# ======================================================================
def ellipse_to_cov_EN(a_major_as, b_minor_as, angle_deg):
    """Localization ellipse -> 2x2 on-sky covariance in (East, North), arcsec^2.
    a_major, b_minor in arcsec (already scaled to 1-sigma); angle = PA of the
    major axis in degrees East of North."""
    th = math.radians(angle_deg)
    e_major = np.array([math.sin(th), math.cos(th)])   # (East, North)
    e_minor = np.array([math.cos(th), -math.sin(th)])
    return (a_major_as ** 2) * np.outer(e_major, e_major) + \
           (b_minor_as ** 2) * np.outer(e_minor, e_minor)


def build_equatorial_cov(row, dec_deg):
    """Return (C_EN arcsec^2, origin_str) or (None, None).
    Priority: explicit a/b/angle ellipse > arcsec errors > time-second RA error."""
    a_err = to_float(row.get("a_err"))
    b_err = to_float(row.get("ell_b_err"))
    angle = to_float(row.get("angle"))
    if a_err is not None and b_err is not None and angle is not None:
        a_as = a_err * 60.0 / ELLIPSE_NSIGMA       # arcmin -> arcsec, to 1-sigma
        b_as = b_err * 60.0 / ELLIPSE_NSIGMA
        return ellipse_to_cov_EN(a_as, b_as, angle), "from_ellipse"

    # diagonal covariance from separate RA / Dec errors (all treated as on-sky)
    sigE = None  # East = on-sky RA error (arcsec)
    sigN = None  # North = Dec error (arcsec)
    ra_arcsec = to_float(row.get("ra_err_arcsec"))
    ra_sec    = to_float(row.get("ra_err_sec"))
    dec_arcsec = to_float(row.get("dec_err_arcsec"))
    if ra_arcsec is not None:
        sigE = ra_arcsec                                    # already on-sky
    elif ra_sec is not None:
        sigE = ra_sec * 15.0 * math.cos(math.radians(dec_deg))  # time-sec -> on-sky arcsec
    if dec_arcsec is not None:
        sigN = dec_arcsec
    if sigE is None and sigN is None:
        return None, None
    # if only one axis known, use it for that axis and leave the other ~0
    cE = (sigE or 0.0) ** 2
    cN = (sigN or 0.0) ** 2
    return np.array([[cE, 0.0], [0.0, cN]]), "measured"


def eq_to_gal_jacobian(ra_deg, dec_deg, delta_as=1.0):
    """Numerical 2x2 Jacobian mapping on-sky equatorial (E,N) offsets to on-sky
    Galactic (E=l*cos b, N=b) offsets at this point. It is a pure rotation, so it
    correctly rotates the error covariance into the Galactic frame."""
    d = delta_as / 3600.0  # deg
    l0, b0 = radec_to_lb(ra_deg, dec_deg)
    cosd = math.cos(math.radians(dec_deg))
    cosb = math.cos(math.radians(b0))

    def gal_offset(ra2, dec2):
        l2, b2 = radec_to_lb(ra2, dec2)
        dl = ((l2 - l0 + 540.0) % 360.0) - 180.0   # wrap-safe
        return np.array([dl * cosb * 3600.0, (b2 - b0) * 3600.0])  # arcsec (E,N)_gal

    col_E = gal_offset(ra_deg + d / max(cosd, 1e-8), dec_deg) / delta_as
    col_N = gal_offset(ra_deg, dec_deg + d) / delta_as
    return np.column_stack([col_E, col_N])


def equatorial_err_to_lb(ra_deg, dec_deg, C_EN):
    """Rotate equatorial covariance into Galactic frame; return on-sky (l_err, b_err) arcsec."""
    J = eq_to_gal_jacobian(ra_deg, dec_deg)
    C_gal = J @ C_EN @ J.T
    l_err = math.sqrt(max(C_gal[0, 0], 0.0))
    b_err = math.sqrt(max(C_gal[1, 1], 0.0))
    return l_err, b_err


# ======================================================================
# Milky Way DM via NE2025
# ======================================================================
def mw_dm_ne2025(l_deg, b_deg):
    """Integrated Galactic (disk) DM toward (l,b) using NE2025. pc/cm^3."""
    Dk, Dv, Du, Dd = ne2025(ldeg=float(l_deg), bdeg=float(b_deg),
                            dmd=NE2025_DIST_KPC, ndir=-1,
                            classic=False, dmd_only=True)
    if NE2025_DM_KEY in Dv:
        return float(Dv[NE2025_DM_KEY])
    for k in Dv:  # fallback: find an exact 'DM' key
        if str(k).strip().upper() == "DM":
            return float(Dv[k])
    raise KeyError("Could not find the DM key in NE2025 output; keys were: %s. "
                   "Set NE2025_DM_KEY in the config." % list(Dv.keys()))


# ======================================================================
# Core per-burst computation (works on a plain dict)
# ======================================================================
def derive_burst(row):
    """row: dict of available fields (logical names). Returns dict of derived fields."""
    out = {}
    notes = []

    # ---- 1. positions -------------------------------------------------
    l_in = to_float(row.get("l_deg"))
    b_in = to_float(row.get("b_deg"))
    ra_in = parse_ra_hms(row.get("ra_hms"))
    if ra_in is None:
        ra_in = to_float(row.get("ra_deg"))
    dec_in = parse_dec_dms(row.get("dec_dms"))
    if dec_in is None:
        dec_in = to_float(row.get("dec_deg"))

    ra = dec = l = b = None
    radec_origin = lb_origin = "none"

    have_lb = (l_in is not None and b_in is not None)
    have_rd = (ra_in is not None and dec_in is not None)

    if not _HAVE_ASTROPY and (have_lb or have_rd):
        notes.append("astropy not installed: coordinate conversion skipped")
    else:
        if have_rd:
            ra, dec = ra_in, dec_in
            radec_origin = "measured_sexagesimal"
            if have_lb:
                l, b = l_in, b_in
                lb_origin = "measured"
            else:
                l, b = radec_to_lb(ra, dec)
                lb_origin = "from_radec"
        elif have_lb:
            l, b = l_in, b_in
            lb_origin = "measured"
            ra, dec = lb_to_radec(l, b)
            radec_origin = "from_lb"
        else:
            notes.append("no usable position")

    if ra is not None:
        ra_hms_o, dec_dms_o = fmt_hmsdms(ra, dec)
        out.update(ra_deg_calc=round(ra, 6), dec_deg_calc=round(dec, 6),
                   ra_hms_calc=ra_hms_o, dec_dms_calc=dec_dms_o,
                   l_deg_calc=round(l, 6), b_deg_calc=round(b, 6))
    out["radec_origin"] = radec_origin
    out["lb_origin"] = lb_origin

    # ---- 2. positional errors ----------------------------------------
    l_err = b_err = ra_err_o = dec_err_o = None
    radec_err_origin = lb_err_origin = "none"
    if ra is not None and _HAVE_ASTROPY:
        C_EN, origin = build_equatorial_cov(row, dec)
        if C_EN is not None:
            ra_err_o = math.sqrt(max(C_EN[0, 0], 0.0))   # on-sky arcsec
            dec_err_o = math.sqrt(max(C_EN[1, 1], 0.0))
            radec_err_origin = origin
            try:
                l_err, b_err = equatorial_err_to_lb(ra, dec, C_EN)
                lb_err_origin = "converted_from_radec"
            except Exception as e:
                notes.append("lb error conversion failed: %s" % e)
    out.update(
        ra_err_arcsec_calc=None if ra_err_o is None else round(ra_err_o, 4),
        dec_err_arcsec_calc=None if dec_err_o is None else round(dec_err_o, 4),
        radec_err_origin=radec_err_origin,
        l_err_arcsec_calc=None if l_err is None else round(l_err, 4),
        b_err_arcsec_calc=None if b_err is None else round(b_err, 4),
        lb_err_origin=lb_err_origin,
    )

    # ---- 3. Milky Way DM + extragalactic DM --------------------------
    dm_obs = to_float(row.get("dm_obs"))
    dm_obs_err = to_float(row.get("dm_obs_err"))
    dm_mw = dm_mw_err = dm_exgal = dm_exgal_err = None
    dm_mw_origin = "none"

    if l is not None:
        if _HAVE_MWPROP:
            try:
                dm_mw = mw_dm_ne2025(l, b)
                dm_mw_err = MW_FRAC_ERR * dm_mw
                dm_mw_origin = "NE2025"
            except Exception as e:
                notes.append("NE2025 failed: %s" % e)
        else:
            notes.append("mwprop not installed: dm_milky_way skipped")

    if dm_mw is not None and dm_obs is not None:
        dm_exgal = dm_obs - dm_mw
        e2 = (dm_obs_err or 0.0) ** 2 + (dm_mw_err or 0.0) ** 2
        dm_exgal_err = math.sqrt(e2)
        if dm_exgal < 0:
            notes.append("dm_exgal < 0 (MW model DM exceeds observed DM)")

    out.update(
        dm_milky_way_calc=None if dm_mw is None else round(dm_mw, 3),
        dm_milky_way_err_calc=None if dm_mw_err is None else round(dm_mw_err, 3),
        dm_mw_origin=dm_mw_origin,
        dm_exgal_calc=None if dm_exgal is None else round(dm_exgal, 3),
        dm_exgal_err_calc=None if dm_exgal_err is None else round(dm_exgal_err, 3),
        dm_halo_calc=DM_HALO,
        notes_calc="; ".join(notes),
    )
    return out


# ======================================================================
# CSV column resolution (handles the duplicate 'b_err' header)
# ======================================================================
def _norm(h):
    return re.sub(r"\s+", " ", str(h).strip().strip('"').strip()).lower()

_ACCEPT = {
    "frb":           ["frb", "frb name", "name", "burst"],
    "l_deg":         ["l (deg)", "l_deg", "l deg", "gl"],
    "b_deg":         ["b (deg)", "b_deg", "b deg", "gb"],
    "l_err":         ["l_err", "l err"],
    "ra_hms":        ["ra (hms)", "ra_hms", "ra hms", "ra"],
    "ra_deg":        ["ra (deg)", "ra_deg", "ra deg"],
    "dec_dms":       ["dec (dms)", "dec_dms", "dec dms", "dec"],
    "dec_deg":       ["dec (deg)", "dec_deg", "dec deg"],
    "ra_err_sec":    ["ra_err_sec", "ra err sec"],
    "ra_err_arcsec": ["ra_err_arcsec", "ra err arcsec"],
    "dec_err_arcsec":["dec_err ('')", "dec_err", "dec err", "dec_err_arcsec", "dec_err (arcsec)"],
    "a_err":         ["a_err", "a err", "semi_major", "semimajor"],
    "angle":         ["angle", "pa", "position angle", "position_angle"],
    "dm_obs":        ["dm obs (pc cm^-3)", "dm_obs", "dm obs", "dm"],
    "dm_obs_err":    ["dm obs error bar", "dm_obs_err", "dm obs err", "dm error", "dm obs error"],
    "z":             ["z", "redshift"],
}

def resolve_columns(header):
    """Map logical field -> column index. Disambiguates the two 'b_err' columns."""
    norm = [_norm(h) for h in header]
    idx = {}
    for logical, names in _ACCEPT.items():
        for i, h in enumerate(norm):
            if h in names:
                idx[logical] = i
                break
        else:
            idx[logical] = None

    # the two b_err columns: Galactic (near l/b) vs ellipse semi-minor (near a_err)
    b_idxs = [i for i, h in enumerate(norm) if h == "b_err"]
    a_idx = idx.get("a_err")
    if len(b_idxs) >= 2:
        ell_b = min(b_idxs, key=lambda i: abs(i - a_idx)) if a_idx is not None else b_idxs[1]
        gal_b = [i for i in b_idxs if i != ell_b][0]
        idx["b_err_gal"], idx["ell_b_err"] = gal_b, ell_b
    elif len(b_idxs) == 1:
        i = b_idxs[0]
        if a_idx is not None and abs(i - a_idx) <= 2:
            idx["ell_b_err"], idx["b_err_gal"] = i, None
        else:
            idx["b_err_gal"], idx["ell_b_err"] = i, None
    else:
        idx["b_err_gal"] = idx["ell_b_err"] = None
    return idx


# ======================================================================
# Top-level: process a CSV
# ======================================================================
def process_csv(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not _HAVE_ASTROPY:
        print("WARNING: astropy not found -> coordinate/error conversion will be skipped. "
              "Install with: pip install astropy")
    if not _HAVE_MWPROP:
        print("WARNING: mwprop not found -> Milky Way / extragalactic DM will be skipped. "
              "Install with: pip install mwprop")

    # read raw, keep duplicate headers distinct via index access
    raw = pd.read_csv(input_csv, header=0, dtype=str, keep_default_na=False)
    header = list(raw.columns)
    cidx = resolve_columns(header)

    def cell(r, logical):
        i = cidx.get(logical)
        return None if i is None else raw.iat[r, i]

    # drop annotation rows that have no FRB name
    frb_i = cidx.get("frb", 0)
    keep_rows = [r for r in range(len(raw)) if clean(raw.iat[r, frb_i]) is not None]

    derived_records = []
    for r in keep_rows:
        logical_row = {
            "l_deg": cell(r, "l_deg"), "b_deg": cell(r, "b_deg"),
            "ra_hms": cell(r, "ra_hms"), "ra_deg": cell(r, "ra_deg"),
            "dec_dms": cell(r, "dec_dms"), "dec_deg": cell(r, "dec_deg"),
            "ra_err_sec": cell(r, "ra_err_sec"), "ra_err_arcsec": cell(r, "ra_err_arcsec"),
            "dec_err_arcsec": cell(r, "dec_err_arcsec"),
            "a_err": cell(r, "a_err"), "ell_b_err": cell(r, "ell_b_err"),
            "angle": cell(r, "angle"),
            "dm_obs": cell(r, "dm_obs"), "dm_obs_err": cell(r, "dm_obs_err"),
        }
        derived_records.append(derive_burst(logical_row))

    derived = pd.DataFrame(derived_records, index=keep_rows)
    original = raw.loc[keep_rows].reset_index(drop=True)
    derived = derived.reset_index(drop=True)
    result = pd.concat([original, derived], axis=1)

    if output_csv is None:
        stem, ext = os.path.splitext(input_csv)
        output_csv = stem + "_processed.csv"
    result.to_csv(output_csv, index=False)
    print(f"Wrote {len(result)} bursts -> {output_csv}")
    n_dm = derived["dm_exgal_calc"].notna().sum() if "dm_exgal_calc" in derived else 0
    print(f"  dm_exgal computed for {n_dm} bursts")
    return result


# ======================================================================
# Manual entry for a single unpublished burst (same core logic)
# ======================================================================
def manual_burst(**fields):
    """Pass any subset of logical fields, e.g.:
        manual_burst(ra_hms='05h31m58s', dec_dms='+33d08m04s',
                     dm_obs=520, dm_obs_err=2,
                     a_err=0.5, ell_b_err=0.3, angle=45)   # a/b in arcmin, angle deg E of N
    Returns the derived dict.
    """
    return derive_burst(fields)


if __name__ == "__main__":
    process_csv()