#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frb_catalogue.py  --  Nexus 2026 FRB catalogue filler
=====================================================
Reads the Nexus 2026 catalogue CSV, fills in every blank it can derive, and
writes a new CSV.  Your input file is never modified.

Matches the current 31-column sheet:
    FRB, <flag>, l_deg, l_err, b_deg, b_err, ra_hms, ra_deg, ra_err_sec,
    ra_err_arcsec, dec_dms, dec_deg, dec_err, dm_obs, dm_err_pos, dm_err_neg,
    dm_ne2001, z, z_err, scatt_tau_ms, scatt_freq_ghz, scattering_err_up,
    scattering_err_down, repeater? y/n, dm_ne2025_2dec, dm_mwhalo, dm_exgal,
    position/dm ref, redshift ref, scattering ref, random notes

WHAT IT DOES, per burst
-----------------------
 1. Parses sexagesimal RA/Dec (handles  h m s / hh mm min ss / ° d ′ ' ″ '' "
    and stray spaces) -> decimal degrees.  Fills l/b from RA/Dec, or RA/Dec
    from l/b, whichever is missing.
 2. Computes the Milky Way *disk* DM toward the burst with NE2025 (mwprop),
    and writes it to dm_ne2025_2dec (2 decimal places, as the column name says).
 3. dm_exgal = dm_obs - dm_disk(NE2025).   The halo is NOT subtracted; it is
    recorded separately in dm_mwhalo.  (This reproduces the convention already
    present in your sheet -- see VERIFIED CONVENTIONS below.)
 4. Propagates position errors.  ra_err_sec / ra_err_arcsec / dec_err are
    rotated through the ICRS->Galactic frame to give l_err / b_err.
 5. Reads a 90% localisation ellipse out of the free-text `random notes`
    column (FRBs 20220509G, 20220914A) and converts it to 1-sigma
    ra_err_arcsec / dec_err.

FILL POLICY
-----------
Existing values are NEVER overwritten.  A blank cell (empty, '..', 'n/a', ...)
gets filled; a populated cell is left exactly as you typed it.  In addition,
every derived quantity is ALSO written to a sibling `*_calc` column so you can
diff the computed value against the value you already had, and a `*_origin`
column records where each filled value came from.

Set FILL_IN_PLACE = False if you want the original columns untouched and only
the _calc columns appended.

----------------------------------------------------------------------
SETUP:   pip install mwprop pandas astropy numpy
USAGE:   python frb_catalogue.py
         (or)  from frb_catalogue import process_csv; process_csv("myfile.csv")
----------------------------------------------------------------------
"""

import os
import re
import math
import numpy as np
import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
INPUT_CSV = "DM_Z_GRAPHING_FRBS (1).csv"
OUTPUT_CSV = None                 # None -> "<input>_processed.csv"

FILL_IN_PLACE = True              # fill blanks in your existing columns
ADD_CALC_COLUMNS = False           # also append *_calc / *_origin columns

MW_FRAC_ERR = 0.20                # fractional uncertainty assumed on the NE2025 disk DM
DM_HALO = 30.0                    # pc/cm^3 -- matches the dm_mwhalo column in your sheet.
                                  # Recorded only; NOT subtracted from dm_exgal.
SUBTRACT_HALO_FROM_EXGAL = False  # keep False to match your sheet's existing convention

NE2025_DIST_KPC = 100.0           # integrate electron density out to here
NE2025_DM_KEY = "DM"              # key NE2025 returns the integrated DM under

DM_NE2025_DECIMALS = 2            # column is called dm_ne2025_2dec

# --- 90% error ellipse -> 1 sigma -------------------------------------
# A 2D 90% confidence ellipse has semi-axes k*sigma with k = sqrt(chi2_2dof(0.90))
# = 2.1460.  If a paper instead quotes each axis as an independent 1D 90%
# interval, the right factor is 1.6449.  See CLARIFY #1 in the chat.
ELLIPSE_90_TO_1SIGMA = 2.1460
NOTES_AXES_ARE_SEMI_AXES = True   # True: quoted numbers are semi-axes (half-widths)
                                  # False: they are full axis lengths -> halved

# Strings that count as "empty" in your sheet
MISSING = {"", "n/a", "na", "..", "...", ".", "?", "nan", "none", "-", "—", "–"}


# ======================================================================
# Optional dependencies
# ======================================================================
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
# Parsing helpers
# ======================================================================
def clean(s):
    """Stripped string, or None if it is a missing-value marker."""
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


def is_blank(s):
    return clean(s) is None


def _fmt_cell(v):
    """Format a derived value for writing back into the dtype=str frame.

    Numbers are already rounded in derive_burst, so print them in full: "%g"
    would truncate to 6 significant digits and throw away sub-arcsec precision
    in l_deg / b_deg, and repr() on a numpy scalar leaks 'np.float64(...)'.
    """
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    return str(v)


def parse_ra_hms(s):
    """'22h15m37.712s', '01h58min00.75017s', ' 04h55m46.96s' -> degrees."""
    t = clean(s)
    if t is None:
        return None
    t = t.replace(" ", "")
    # 'min' must be collapsed before the single-letter 'm' pattern is applied
    t = re.sub(r"min", "m", t, flags=re.I)
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)h(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?$", t, re.I)
    if not m:
        try:
            return float(t)          # maybe already decimal degrees
        except ValueError:
            return None
    h, mi, se = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return 15.0 * (h + mi / 60.0 + se / 3600.0)


def parse_dec_dms(s):
    """Dec with  ° d  /  ' ′  /  '' \" ″  separators -> degrees."""
    t = clean(s)
    if t is None:
        return None
    t = (t.replace("\u2032", "'").replace("\u2019", "'")
          .replace("\u2033", '"').replace("\u201c", '"').replace("\u201d", '"'))
    t = t.replace("\u00b0", "d").replace("\u00ba", "d")
    t = t.replace("''", '"').replace("'", "m").replace('"', "s")
    t = t.replace(" ", "")
    m = re.match(r"^([+-]?)(\d+(?:\.\d+)?)d(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?$", t, re.I)
    if not m:
        try:
            return float(clean(s))
        except (ValueError, TypeError):
            return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    d, mi, se = float(m.group(2)), float(m.group(3)), float(m.group(4))
    return sign * (d + mi / 60.0 + se / 3600.0)


# ----------------------------------------------------------------------
# Free-text ellipse extraction from the `random notes` column
# ----------------------------------------------------------------------
_ELLIPSE_RE = re.compile(
    r"(?P<conf>\d{2})\s*%\s*(?:error\s*|localisation\s*|localization\s*)?ellipse"
    r".*?axes?\s*"
    r"(?P<a>\d+(?:\.\d+)?)\s*(?:''|\"|\u2033|arcsec)?\s*(?:and|x|\u00d7|,)\s*"
    r"(?P<b>\d+(?:\.\d+)?)\s*(?:''|\"|\u2033|arcsec)?"
    r".*?(?P<order>R\.?\s*A\.?|dec)",
    re.I | re.S,
)


def parse_ellipse_from_notes(note):
    """Extract a quoted confidence ellipse from free text.

    Handles e.g.
        "90% error ellipse with axes 4.7'' and 3.2'' in R.A. and decl. respectively"

    Returns (sigma_ra_arcsec, sigma_dec_arcsec, provenance_string) or
    (None, None, None) if nothing parseable is found.

    Assumes the ellipse is axis-aligned with RA/Dec, which is what
    "in R.A. and decl. respectively" states.  A rotated ellipse would need a
    position angle, which this text does not provide.
    """
    t = clean(note)
    if t is None:
        return None, None, None
    m = _ELLIPSE_RE.search(t)
    if not m:
        return None, None, None

    conf = float(m.group("conf"))
    a = float(m.group("a"))
    b = float(m.group("b"))

    # "in R.A. and decl. respectively" -> first number is RA, second is Dec.
    # If the text names dec first, swap.
    if m.group("order").lower().startswith("dec"):
        a, b = b, a

    if not NOTES_AXES_ARE_SEMI_AXES:
        a, b = a / 2.0, b / 2.0

    if abs(conf - 90.0) < 1e-6:
        k = ELLIPSE_90_TO_1SIGMA
    elif abs(conf - 68.0) < 2.0:
        k = 1.0
    elif abs(conf - 95.0) < 1e-6:
        k = 2.4477            # sqrt(chi2_2dof(0.95))
    else:
        return None, None, None

    prov = "from_notes_%d%%_ellipse_k=%.4f" % (int(conf), k)
    return a / k, b / k, prov


# ======================================================================
# Coordinate transforms
# ======================================================================
# Pure-numpy ICRS <-> Galactic rotation, used only if astropy is unavailable.
# Built from the standard Galactic pole / node (J2000): the accuracy is
# ~milliarcsec, which is far below any FRB localisation.
_RA_NGP, _DEC_NGP, _L_NCP = math.radians(192.85948), math.radians(27.12825), math.radians(122.93192)


def _rot_matrix():
    sa, ca = math.sin(_RA_NGP), math.cos(_RA_NGP)
    sd, cd = math.sin(_DEC_NGP), math.cos(_DEC_NGP)
    sl, cl = math.sin(_L_NCP), math.cos(_L_NCP)
    R_z_ra = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])
    R_y = np.array([[sd, 0, -cd], [0, 1, 0], [cd, 0, sd]])
    R_z_l = np.array([[cl, sl, 0], [-sl, cl, 0], [0, 0, 1]])
    return R_z_l @ R_y @ R_z_ra


_R_EQ2GAL = _rot_matrix()


def _sph2vec(lon_deg, lat_deg):
    lo, la = math.radians(lon_deg), math.radians(lat_deg)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _vec2sph(v):
    v = v / np.linalg.norm(v)
    lon = math.degrees(math.atan2(v[1], v[0])) % 360.0
    lat = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
    return lon, lat


def radec_to_lb(ra_deg, dec_deg):
    if _HAVE_ASTROPY:
        g = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
        return float(g.l.deg), float(g.b.deg)
    return _vec2sph(_R_EQ2GAL @ _sph2vec(ra_deg, dec_deg))


def lb_to_radec(l_deg, b_deg):
    if _HAVE_ASTROPY:
        i = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
        return float(i.ra.deg), float(i.dec.deg)
    return _vec2sph(_R_EQ2GAL.T @ _sph2vec(l_deg, b_deg))


def fmt_hmsdms(ra_deg, dec_deg):
    """Format back to the sexagesimal style used in the sheet."""
    if _HAVE_ASTROPY:
        c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        ra_str = c.ra.to_string(unit=u.hour, sep="hms", precision=3, pad=True)
        dec_str = c.dec.to_string(unit=u.deg, sep="dms", precision=2,
                                  alwayssign=True, pad=True)
        return ra_str, dec_str
    h_tot = ra_deg / 15.0
    h = int(h_tot); mi = int((h_tot - h) * 60); se = (h_tot - h - mi / 60.0) * 3600.0
    sign = "-" if dec_deg < 0 else "+"
    ad = abs(dec_deg)
    d = int(ad); am = int((ad - d) * 60); asec = (ad - d - am / 60.0) * 3600.0
    return ("%02dh%02dm%06.3fs" % (h, mi, se),
            "%s%02dd%02dm%05.2fs" % (sign, d, am, asec))


# ======================================================================
# Positional errors
# ======================================================================
def build_equatorial_cov(row, dec_deg):
    """Return (2x2 on-sky covariance in (East, North) arcsec^2, origin) or (None, None).

    Priority:  explicit arcsec errors  >  ellipse parsed from notes  >  RA error
    in time-seconds.  Errors are treated as on-sky angles; the time-second RA
    error is converted with the cos(dec) factor.
    """
    sigE = sigN = None
    origins = []

    ra_arcsec = to_float(row.get("ra_err_arcsec"))
    ra_sec = to_float(row.get("ra_err_sec"))
    dec_arcsec = to_float(row.get("dec_err"))

    if ra_arcsec is not None:
        sigE = ra_arcsec
        origins.append("ra_err_arcsec")
    elif ra_sec is not None:
        sigE = ra_sec * 15.0 * math.cos(math.radians(dec_deg))
        origins.append("ra_err_sec*15*cos(dec)")
    if dec_arcsec is not None:
        sigN = dec_arcsec
        origins.append("dec_err")

    # fall back to an ellipse quoted in the notes
    if sigE is None or sigN is None:
        e_ra, e_dec, prov = parse_ellipse_from_notes(row.get("notes"))
        if e_ra is not None:
            if sigE is None:
                sigE = e_ra
                origins.append(prov)
            if sigN is None:
                sigN = e_dec
                if prov not in origins:
                    origins.append(prov)

    if sigE is None and sigN is None:
        return None, None
    cE = (sigE or 0.0) ** 2
    cN = (sigN or 0.0) ** 2
    return np.array([[cE, 0.0], [0.0, cN]]), "+".join(origins)


def eq_to_gal_jacobian(ra_deg, dec_deg, delta_as=1.0):
    """Numerical 2x2 Jacobian mapping on-sky equatorial (E,N) offsets to on-sky
    Galactic (E = l*cos b, N = b) offsets.  It is a pure rotation, so it
    correctly rotates the covariance into the Galactic frame."""
    d = delta_as / 3600.0
    l0, b0 = radec_to_lb(ra_deg, dec_deg)
    cosd = math.cos(math.radians(dec_deg))
    cosb = math.cos(math.radians(b0))

    def gal_offset(ra2, dec2):
        l2, b2 = radec_to_lb(ra2, dec2)
        dl = ((l2 - l0 + 540.0) % 360.0) - 180.0
        return np.array([dl * cosb * 3600.0, (b2 - b0) * 3600.0])

    col_E = gal_offset(ra_deg + d / max(cosd, 1e-8), dec_deg) / delta_as
    col_N = gal_offset(ra_deg, dec_deg + d) / delta_as
    return np.column_stack([col_E, col_N])


def equatorial_err_to_lb(ra_deg, dec_deg, C_EN):
    """Rotate the equatorial covariance into the Galactic frame.
    Returns on-sky (l_err, b_err) in arcsec."""
    J = eq_to_gal_jacobian(ra_deg, dec_deg)
    C_gal = J @ C_EN @ J.T
    return math.sqrt(max(C_gal[0, 0], 0.0)), math.sqrt(max(C_gal[1, 1], 0.0))


# ======================================================================
# Milky Way DM via NE2025
# ======================================================================
def mw_dm_ne2025(l_deg, b_deg):
    """Integrated Galactic disk DM toward (l,b), pc/cm^3."""
    Dk, Dv, Du, Dd = ne2025(ldeg=float(l_deg), bdeg=float(b_deg),
                            dmd=NE2025_DIST_KPC, ndir=-1,
                            classic=False, dmd_only=True)
    if NE2025_DM_KEY in Dv:
        return float(Dv[NE2025_DM_KEY])
    for k in Dv:
        if str(k).strip().upper() == "DM":
            return float(Dv[k])
    raise KeyError("No DM key in NE2025 output; keys were: %s. "
                   "Set NE2025_DM_KEY." % list(Dv.keys()))


# ======================================================================
# Core per-burst computation
# ======================================================================
def derive_burst(row):
    """row: dict of logical field names -> raw cell strings.
    Returns a dict of derived values (suffix-free logical names)."""
    out = {}
    notes = []

    # ---- 1. position -------------------------------------------------
    l_in, b_in = to_float(row.get("l_deg")), to_float(row.get("b_deg"))
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

    if have_rd:
        ra, dec = ra_in, dec_in
        radec_origin = "measured"
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

    # consistency check: sexagesimal vs decimal columns should agree
    ra_dec_col = to_float(row.get("ra_deg"))
    dec_dec_col = to_float(row.get("dec_deg"))
    ra_sex = parse_ra_hms(row.get("ra_hms"))
    dec_sex = parse_dec_dms(row.get("dec_dms"))
    if ra_sex is not None and ra_dec_col is not None and abs(ra_sex - ra_dec_col) > 0.01:
        notes.append("ra_hms and ra_deg disagree by %.4f deg" % abs(ra_sex - ra_dec_col))
    if dec_sex is not None and dec_dec_col is not None and abs(dec_sex - dec_dec_col) > 0.01:
        notes.append("dec_dms and dec_deg disagree by %.4f deg" % abs(dec_sex - dec_dec_col))

    if ra is not None:
        ra_hms_o, dec_dms_o = fmt_hmsdms(ra, dec)
        out.update(ra_deg=round(ra, 6), dec_deg=round(dec, 6),
                   ra_hms=ra_hms_o, dec_dms=dec_dms_o,
                   l_deg=round(l, 6), b_deg=round(b, 6))
    out["radec_origin"] = radec_origin
    out["lb_origin"] = lb_origin

    # ---- 2. positional errors ---------------------------------------
    l_err = b_err = ra_err_o = dec_err_o = None
    radec_err_origin = lb_err_origin = "none"
    if ra is not None:
        C_EN, origin = build_equatorial_cov(row, dec)
        if C_EN is not None:
            ra_err_o = math.sqrt(max(C_EN[0, 0], 0.0))
            dec_err_o = math.sqrt(max(C_EN[1, 1], 0.0))
            radec_err_origin = origin
            try:
                l_err, b_err = equatorial_err_to_lb(ra, dec, C_EN)
                lb_err_origin = "rotated_from_radec"
            except Exception as e:
                notes.append("l/b error rotation failed: %s" % e)

    out.update(
        ra_err_arcsec=None if ra_err_o is None else round(ra_err_o, 4),
        dec_err=None if dec_err_o is None else round(dec_err_o, 4),
        radec_err_origin=radec_err_origin,
        l_err=None if l_err is None else round(l_err, 4),
        b_err=None if b_err is None else round(b_err, 4),
        lb_err_origin=lb_err_origin,
    )

    # ---- 3. Milky Way DM + extragalactic DM -------------------------
    dm_obs = to_float(row.get("dm_obs"))
    dm_ep = to_float(row.get("dm_err_pos"))
    dm_en = to_float(row.get("dm_err_neg"))
    dm_mw = dm_mw_err = dm_exgal = None
    dm_exgal_ep = dm_exgal_en = None
    dm_mw_origin = "none"

    if l is not None:
        if _HAVE_MWPROP:
            try:
                dm_mw = mw_dm_ne2025(l, b)
                dm_mw_err = MW_FRAC_ERR * dm_mw
                dm_mw_origin = "NE2025"
            except Exception as e:
                notes.append("NE2025 failed: %s" % e)
        # (if mwprop is absent, that is reported once by process_csv, not per row)

    if dm_mw is not None and dm_obs is not None:
        dm_exgal = dm_obs - dm_mw
        if SUBTRACT_HALO_FROM_EXGAL:
            dm_exgal -= DM_HALO
        # asymmetric observed-DM errors propagated separately, each added in
        # quadrature with the NE2025 model uncertainty
        mw2 = (dm_mw_err or 0.0) ** 2
        dm_exgal_ep = math.sqrt((dm_ep or 0.0) ** 2 + mw2)
        dm_exgal_en = math.sqrt((dm_en or 0.0) ** 2 + mw2)
        if dm_exgal < 0:
            notes.append("dm_exgal < 0 (model MW DM exceeds observed DM)")

    out.update(
        dm_ne2025_2dec=None if dm_mw is None else round(dm_mw, DM_NE2025_DECIMALS),
        dm_ne2025_err=None if dm_mw_err is None else round(dm_mw_err, 3),
        dm_mw_origin=dm_mw_origin,
        dm_mwhalo=DM_HALO,
        dm_exgal=None if dm_exgal is None else round(dm_exgal, 3),
        dm_exgal_err_pos=None if dm_exgal_ep is None else round(dm_exgal_ep, 3),
        dm_exgal_err_neg=None if dm_exgal_en is None else round(dm_exgal_en, 3),
        dm_exgal_origin=("dm_obs - dm_ne2025" if dm_exgal is not None else "none"),
        notes_calc="; ".join(notes),
    )
    return out


# ======================================================================
# Column resolution for the current sheet
# ======================================================================
def _norm(h):
    return re.sub(r"\s+", " ", str(h).strip().strip('"').strip()).lower()


_ACCEPT = {
    "frb":            ["frb", "frb name", "name", "burst"],
    "l_deg":          ["l_deg", "l (deg)", "l deg", "gl"],
    "l_err":          ["l_err", "l err"],
    "b_deg":          ["b_deg", "b (deg)", "b deg", "gb"],
    "b_err":          ["b_err", "b err"],
    "ra_hms":         ["ra_hms", "ra (hms)", "ra hms", "ra"],
    "ra_deg":         ["ra_deg", "ra (deg)", "ra deg"],
    "ra_err_sec":     ["ra_err_sec", "ra err sec"],
    "ra_err_arcsec":  ["ra_err_arcsec", "ra err arcsec"],
    "dec_dms":        ["dec_dms", "dec (dms)", "dec dms", "dec"],
    "dec_deg":        ["dec_deg", "dec (deg)", "dec deg"],
    "dec_err":        ["dec_err", "dec err", "dec_err ('')", "dec_err_arcsec"],
    "dm_obs":         ["dm_obs", "dm obs", "dm", "dm obs (pc cm^-3)"],
    "dm_err_pos":     ["dm_err_pos", "dm err pos", "dm_err_up"],
    "dm_err_neg":     ["dm_err_neg", "dm err neg", "dm_err_down"],
    "dm_ne2001":      ["dm_ne2001", "dm ne2001"],
    "z":              ["z", "redshift"],
    "z_err":          ["z_err", "z err"],
    "dm_ne2025_2dec": ["dm_ne2025_2dec", "dm_ne2025", "dm ne2025"],
    "dm_mwhalo":      ["dm_mwhalo", "dm mwhalo", "dm_halo", "dm halo"],
    "dm_exgal":       ["dm_exgal", "dm exgal"],
    "notes":          ["random notes", "notes", "comments"],
}

# logical name -> (sheet column it fills, rounding)
_FILL_MAP = [
    ("l_deg", "l_deg"),
    ("b_deg", "b_deg"),
    ("ra_deg", "ra_deg"),
    ("dec_deg", "dec_deg"),
    ("ra_hms", "ra_hms"),
    ("dec_dms", "dec_dms"),
    ("l_err", "l_err"),
    ("b_err", "b_err"),
    ("ra_err_arcsec", "ra_err_arcsec"),
    ("dec_err", "dec_err"),
    ("dm_ne2025_2dec", "dm_ne2025_2dec"),
    ("dm_mwhalo", "dm_mwhalo"),
    ("dm_exgal", "dm_exgal"),
]


def resolve_columns(header):
    """Map logical field -> column index, by exact normalised header match."""
    norm = [_norm(h) for h in header]
    idx = {}
    for logical, names in _ACCEPT.items():
        idx[logical] = None
        for i, h in enumerate(norm):
            if h in names:
                idx[logical] = i
                break
    return idx


# ======================================================================
# Top level
# ======================================================================
def process_csv(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV, verbose=True):
    if not _HAVE_ASTROPY:
        print("NOTE: astropy not found -> using the built-in rotation matrix "
              "(accuracy ~mas, fine for FRB localisations). pip install astropy to silence.")
    if not _HAVE_MWPROP:
        print("WARNING: mwprop not found -> dm_ne2025_2dec and dm_exgal will NOT be "
              "computed. Install with: pip install mwprop")

    raw = pd.read_csv(input_csv, header=0, dtype=str, keep_default_na=False)
    header = list(raw.columns)
    cidx = resolve_columns(header)

    missing_cols = [k for k, v in cidx.items() if v is None]
    if missing_cols and verbose:
        print("Columns not found in sheet (skipped): %s" % ", ".join(missing_cols))

    def cell(r, logical):
        i = cidx.get(logical)
        return None if i is None else raw.iat[r, i]

    frb_i = cidx.get("frb") or 0
    keep_rows = [r for r in range(len(raw)) if clean(raw.iat[r, frb_i]) is not None]

    logical_names = ["l_deg", "b_deg", "ra_hms", "ra_deg", "dec_dms", "dec_deg",
                     "ra_err_sec", "ra_err_arcsec", "dec_err",
                     "dm_obs", "dm_err_pos", "dm_err_neg", "notes"]

    derived_records = []
    for r in keep_rows:
        logical_row = {n: cell(r, n) for n in logical_names}
        derived_records.append(derive_burst(logical_row))

    derived = pd.DataFrame(derived_records)
    original = raw.loc[keep_rows].reset_index(drop=True)
    result = original.copy()

    # ---- fill blanks in place, never overwrite -----------------------
    fill_counts = {}
    if FILL_IN_PLACE:
        for logical, sheet_col in _FILL_MAP:
            ci = cidx.get(logical)
            if ci is None or logical not in derived.columns:
                continue
            colname = header[ci]
            n = 0
            origin_vals = []
            for i in range(len(result)):
                cur = result.iat[i, ci]
                new = derived.iloc[i].get(logical)
                if is_blank(cur) and new is not None and not (
                        isinstance(new, float) and math.isnan(new)):
                    result.iat[i, ci] = _fmt_cell(new)
                    origin_vals.append("computed")
                    n += 1
                elif not is_blank(cur):
                    origin_vals.append("original")
                else:
                    origin_vals.append("unfilled")
            fill_counts[colname] = n
            if ADD_CALC_COLUMNS:
                result[colname + "_origin"] = origin_vals

    # ---- append the _calc columns ------------------------------------
    if ADD_CALC_COLUMNS:
        for logical in derived.columns:
            newname = logical + ("_calc" if logical in dict(_FILL_MAP) else "")
            if newname in result.columns:
                newname = logical + "_calc2"
            result[newname] = derived[logical].values

    if output_csv is None:
        stem, _ = os.path.splitext(input_csv)
        output_csv = stem + "_processed.csv"
    result.to_csv(output_csv, index=False)

    if verbose:
        print("\nWrote %d bursts -> %s" % (len(result), output_csv))
        if fill_counts:
            print("Blanks filled:")
            for k, v in fill_counts.items():
                if v:
                    print("   %-18s %3d" % (k, v))
        flagged = derived["notes_calc"].astype(str).str.len() > 0
        if flagged.any():
            print("\nRows with warnings (%d):" % flagged.sum())
            for i in np.where(flagged.values)[0]:
                print("   %-12s %s" % (result.iloc[i, frb_i], derived.iloc[i]["notes_calc"]))
    return result


def manual_burst(**fields):
    """Derive a single unpublished burst, e.g.
        manual_burst(ra_hms='05h31m58s', dec_dms='+33d08m04s',
                     dm_obs=520, dm_err_pos=2, dm_err_neg=2,
                     ra_err_arcsec=0.5, dec_err=0.3)
    """
    return derive_burst(fields)


if __name__ == "__main__":
    process_csv()