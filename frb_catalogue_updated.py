#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frb_catalogue_updated.py
========================
Process the current FRB catalogue CSV and write a new processed CSV while
leaving the input file unchanged.

The script is matched to these current column names:
    FRB, l_deg, l_err, b_deg, b_err,
    ra_hms, ra_deg, ra_err_sec, ra_err_arcsec,
    dec_dms, dec_deg, dec_err,
    dm_obs, dm_err_pos, dm_err_neg,
    dm_ne2025_2dec, dm_mwhalo, dm_exgal, ...

There is no a_err / ellipse-b_err / angle input. Positional uncertainties are
handled using separate R.A. and declination errors only.

Per burst, the script:
  1. Parses sexagesimal R.A./declination and converts coordinates between ICRS
     and Galactic frames when one frame is missing.
  2. Uses measured l_err/b_err values when they are present. Otherwise, it
     rotates the R.A./declination error covariance into Galactic coordinates.
  3. Calculates the Milky Way disk DM with NE2025.
  4. Calculates dm_exgal = dm_obs - dm_ne2025. The halo value is recorded but
     is not subtracted, matching the existing catalogue convention.
  5. Propagates the positive and negative observed-DM errors separately.
  6. Appends *_calc and *_origin columns. Original columns are never overwritten.

Install once in the active FRB environment:
    pip install mwprop pandas astropy numpy

Run directly:
    python frb_catalogue_updated.py

Or import it:
    from frb_catalogue_updated import process_csv
    process_csv("your_catalogue.csv")
"""

import math
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_CSV = "DM_Z_GRAPHING_FRBS.csv"
OUTPUT_CSV = None  # None -> "<input>_processed.csv"

MW_FRAC_ERR = 0.20       # Assumed fractional uncertainty on NE2025 disk DM.
DM_HALO_DEFAULT = 30.0   # Used only when dm_mwhalo is blank; not subtracted.
NE2025_DIST_KPC = 100.0
NE2025_DM_KEY = "DM"

MISSING = {"", "n/a", "na", "..", "...", ".", "?", "nan", "none", "-", "—", "–"}

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
def clean(value: Any) -> Optional[str]:
    """Return a stripped string, or None for a catalogue missing marker."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING else text


def to_float(value: Any) -> Optional[float]:
    text = clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_ra_hms(value: Any) -> Optional[float]:
    """Parse R.A. as decimal degrees or common sexagesimal variants."""
    text = clean(value)
    if text is None:
        return None

    # A bare number is assumed to already be decimal degrees.
    try:
        return float(text)
    except ValueError:
        pass

    t = text.lower().strip()
    t = t.replace("hours", "h").replace("hour", "h")
    t = t.replace("hrs", "h").replace("hr", "h")
    t = t.replace("minutes", "m").replace("minute", "m").replace("min", "m")
    t = t.replace("seconds", "s").replace("second", "s").replace("sec", "s")
    t = re.sub(r"\s+", "", t)

    # Accept HH:MM:SS.s as well as HHhMMmSS.s.
    if ":" in t:
        parts = t.rstrip("s").split(":")
        if len(parts) == 3:
            try:
                h, minute, second = map(float, parts)
                return 15.0 * (h + minute / 60.0 + second / 3600.0)
            except ValueError:
                return None

    match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)h(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?",
        t,
        re.IGNORECASE,
    )
    if not match:
        return None

    h, minute, second = map(float, match.groups())
    return 15.0 * (h + minute / 60.0 + second / 3600.0)


def parse_dec_dms(value: Any) -> Optional[float]:
    """Parse declination as decimal degrees or common DMS variants."""
    text = clean(value)
    if text is None:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    t = text.strip()
    t = (t.replace("′", "'").replace("’", "'").replace("″", '"').replace("“", '"').replace("”", '"'))
    t = t.replace("°", "d").replace("º", "d")
    t = t.replace("''", '"').replace("'", "m").replace('"', "s")
    t = re.sub(r"\s+", "", t)

    if ":" in t:
        sign = -1.0 if t.startswith("-") else 1.0
        parts = t.lstrip("+-").split(":")
        if len(parts) == 3:
            try:
                degree, minute, second = map(float, parts)
                return sign * (degree + minute / 60.0 + second / 3600.0)
            except ValueError:
                return None

    match = re.fullmatch(
        r"([+-]?)(\d+(?:\.\d+)?)d(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?",
        t,
        re.IGNORECASE,
    )
    if not match:
        return None

    sign = -1.0 if match.group(1) == "-" else 1.0
    degree, minute, second = map(float, match.groups()[1:])
    return sign * (degree + minute / 60.0 + second / 3600.0)


# ======================================================================
# Coordinate transforms
# ======================================================================
def radec_to_lb(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g = c.galactic
    return float(g.l.deg), float(g.b.deg)


def lb_to_radec(l_deg: float, b_deg: float) -> tuple[float, float]:
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    i = c.icrs
    return float(i.ra.deg), float(i.dec.deg)


def fmt_hmsdms(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ra_string = c.ra.to_string(unit=u.hour, sep="hms", precision=5, pad=True)
    dec_string = c.dec.to_string(unit=u.deg, sep="dms", precision=4, alwayssign=True, pad=True)
    return ra_string, dec_string


# ======================================================================
# Positional error handling: separate R.A. and declination errors only
# ======================================================================
def build_equatorial_cov(row: dict[str, Any], dec_deg: float):
    """Build a diagonal on-sky covariance in East/North, in arcsec^2.

    ra_err_arcsec is assumed to already be an on-sky angular R.A. error
    (Delta-alpha cos(delta)). When it is absent, ra_err_sec is interpreted as
    seconds of R.A. time and converted using 15*cos(delta).

    dec_err is interpreted as arcseconds.
    """
    ra_arcsec = to_float(row.get("ra_err_arcsec"))
    ra_time_sec = to_float(row.get("ra_err_sec"))
    dec_arcsec = to_float(row.get("dec_err"))

    if ra_arcsec is not None:
        sig_east = ra_arcsec
        ra_origin = "ra_err_arcsec"
    elif ra_time_sec is not None:
        sig_east = ra_time_sec * 15.0 * math.cos(math.radians(dec_deg))
        ra_origin = "ra_err_sec_converted"
    else:
        sig_east = None
        ra_origin = "none"

    sig_north = dec_arcsec

    # Do not silently treat a missing axis as zero; that would understate errors.
    covariance = None
    if sig_east is not None and sig_north is not None:
        covariance = np.array([[sig_east ** 2, 0.0], [0.0, sig_north ** 2]])

    return covariance, sig_east, sig_north, ra_origin


def eq_to_gal_jacobian(ra_deg: float, dec_deg: float, delta_as: float = 1.0) -> np.ndarray:
    """Numerical Jacobian from equatorial East/North to Galactic East/North."""
    d_deg = delta_as / 3600.0
    l0, b0 = radec_to_lb(ra_deg, dec_deg)
    cos_dec = math.cos(math.radians(dec_deg))
    cos_b = math.cos(math.radians(b0))

    def gal_offset(ra2: float, dec2: float) -> np.ndarray:
        l2, b2 = radec_to_lb(ra2, dec2)
        dl = ((l2 - l0 + 540.0) % 360.0) - 180.0
        return np.array([dl * cos_b * 3600.0, (b2 - b0) * 3600.0])

    col_east = gal_offset(ra_deg + d_deg / max(abs(cos_dec), 1e-8), dec_deg) / delta_as
    col_north = gal_offset(ra_deg, dec_deg + d_deg) / delta_as
    return np.column_stack([col_east, col_north])


def equatorial_err_to_lb(ra_deg: float, dec_deg: float, covariance_en: np.ndarray) -> tuple[float, float]:
    """Rotate covariance and return on-sky Galactic l/b errors in arcseconds."""
    jacobian = eq_to_gal_jacobian(ra_deg, dec_deg)
    covariance_gal = jacobian @ covariance_en @ jacobian.T
    l_err_as = math.sqrt(max(covariance_gal[0, 0], 0.0))
    b_err_as = math.sqrt(max(covariance_gal[1, 1], 0.0))
    return l_err_as, b_err_as


# ======================================================================
# Milky Way DM via NE2025
# ======================================================================
def mw_dm_ne2025(l_deg: float, b_deg: float) -> float:
    """Integrated Galactic disk DM toward (l,b), in pc cm^-3."""
    _, values, _, _ = ne2025(
        ldeg=float(l_deg),
        bdeg=float(b_deg),
        dmd=NE2025_DIST_KPC,
        ndir=-1,
        classic=False,
        dmd_only=True,
    )

    if NE2025_DM_KEY in values:
        return float(values[NE2025_DM_KEY])
    for key in values:
        if str(key).strip().upper() == "DM":
            return float(values[key])
    raise KeyError(
        f"Could not find a DM key in NE2025 output. Keys: {list(values.keys())}. "
        "Update NE2025_DM_KEY in the configuration."
    )


# ======================================================================
# Per-burst calculation
# ======================================================================
def derive_burst(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    notes: list[str] = []

    # ---- 1. Positions -------------------------------------------------
    l_input = to_float(row.get("l_deg"))
    b_input = to_float(row.get("b_deg"))

    ra_input = parse_ra_hms(row.get("ra_hms"))
    if ra_input is None:
        ra_input = to_float(row.get("ra_deg"))

    dec_input = parse_dec_dms(row.get("dec_dms"))
    if dec_input is None:
        dec_input = to_float(row.get("dec_deg"))

    have_lb = l_input is not None and b_input is not None
    have_radec = ra_input is not None and dec_input is not None

    ra = dec = l_deg = b_deg = None
    radec_origin = "none"
    lb_origin = "none"

    if not _HAVE_ASTROPY and (have_lb or have_radec):
        # Decimal coordinates can still be copied, but frame conversion/formatting cannot run.
        if have_radec:
            ra, dec = ra_input, dec_input
            radec_origin = "measured"
        if have_lb:
            l_deg, b_deg = l_input, b_input
            lb_origin = "measured"
        notes.append("astropy not installed: coordinate-frame conversion skipped")
    elif have_radec:
        ra, dec = ra_input, dec_input
        radec_origin = "measured"
        if have_lb:
            l_deg, b_deg = l_input, b_input
            lb_origin = "measured"
        else:
            l_deg, b_deg = radec_to_lb(ra, dec)
            lb_origin = "from_radec"
    elif have_lb:
        l_deg, b_deg = l_input, b_input
        lb_origin = "measured"
        ra, dec = lb_to_radec(l_deg, b_deg)
        radec_origin = "from_lb"
    else:
        notes.append("no usable position")

    if ra is not None and dec is not None:
        out["ra_deg_calc"] = round(ra, 8)
        out["dec_deg_calc"] = round(dec, 8)
        if _HAVE_ASTROPY:
            ra_string, dec_string = fmt_hmsdms(ra, dec)
            out["ra_hms_calc"] = ra_string
            out["dec_dms_calc"] = dec_string

    if l_deg is not None and b_deg is not None:
        out["l_deg_calc"] = round(l_deg, 8)
        out["b_deg_calc"] = round(b_deg, 8)

    out["radec_origin"] = radec_origin
    out["lb_origin"] = lb_origin

    # ---- 2. Position errors ------------------------------------------
    measured_l_err_deg = to_float(row.get("l_err"))
    measured_b_err_deg = to_float(row.get("b_err"))

    ra_err_as = dec_err_as = None
    l_err_as = b_err_as = None
    radec_err_origin = "none"
    lb_err_origin = "none"

    if ra is not None and dec is not None:
        covariance, ra_err_as, dec_err_as, ra_error_source = build_equatorial_cov(row, dec)
        if ra_err_as is not None or dec_err_as is not None:
            radec_err_origin = ra_error_source
            if dec_err_as is not None:
                radec_err_origin += "+dec_err"

        # Preserve measured Galactic-coordinate errors when both are supplied.
        if measured_l_err_deg is not None and measured_b_err_deg is not None:
            l_err_as = measured_l_err_deg * 3600.0
            b_err_as = measured_b_err_deg * 3600.0
            lb_err_origin = "measured_l_err_b_err"
        elif covariance is not None and _HAVE_ASTROPY:
            try:
                l_err_as, b_err_as = equatorial_err_to_lb(ra, dec, covariance)
                lb_err_origin = "converted_from_radec"
            except Exception as exc:
                notes.append(f"l/b error conversion failed: {exc}")
        elif covariance is None and (ra_err_as is not None or dec_err_as is not None):
            notes.append("both R.A. and declination errors are required to derive l/b errors")

    out.update(
        ra_err_arcsec_calc=None if ra_err_as is None else round(ra_err_as, 6),
        dec_err_arcsec_calc=None if dec_err_as is None else round(dec_err_as, 6),
        radec_err_origin=radec_err_origin,
        l_err_calc=None if l_err_as is None else round(l_err_as / 3600.0, 9),
        b_err_calc=None if b_err_as is None else round(b_err_as / 3600.0, 9),
        l_err_arcsec_calc=None if l_err_as is None else round(l_err_as, 6),
        b_err_arcsec_calc=None if b_err_as is None else round(b_err_as, 6),
        lb_err_origin=lb_err_origin,
    )

    # ---- 3. Milky Way and extragalactic DM ---------------------------
    dm_obs = to_float(row.get("dm_obs"))
    dm_err_pos = to_float(row.get("dm_err_pos"))
    dm_err_neg = to_float(row.get("dm_err_neg"))
    halo_dm = to_float(row.get("dm_mwhalo"))
    if halo_dm is None:
        halo_dm = DM_HALO_DEFAULT

    dm_mw = dm_mw_err = dm_exgal = None
    dm_exgal_err_pos = dm_exgal_err_neg = None
    dm_mw_origin = "none"

    # Prefer a fresh NE2025 calculation. If mwprop is unavailable or fails, use
    # an existing dm_ne2025_2dec value when the catalogue already contains one.
    if l_deg is not None and b_deg is not None and _HAVE_MWPROP:
        try:
            dm_mw = mw_dm_ne2025(l_deg, b_deg)
            dm_mw_origin = "NE2025_calculated"
        except Exception as exc:
            notes.append(f"NE2025 failed: {exc}")

    if dm_mw is None:
        existing_dm_mw = to_float(row.get("dm_ne2025_2dec"))
        if existing_dm_mw is not None:
            dm_mw = existing_dm_mw
            dm_mw_origin = "existing_dm_ne2025_2dec"
        elif not _HAVE_MWPROP:
            notes.append("mwprop not installed: NE2025 DM skipped")

    if dm_mw is not None:
        dm_mw_err = MW_FRAC_ERR * dm_mw

    if dm_obs is not None and dm_mw is not None:
        dm_exgal = dm_obs - dm_mw
        if dm_err_pos is not None:
            dm_exgal_err_pos = math.sqrt(dm_err_pos ** 2 + dm_mw_err ** 2)
        if dm_err_neg is not None:
            dm_exgal_err_neg = math.sqrt(dm_err_neg ** 2 + dm_mw_err ** 2)
        if dm_exgal < 0:
            notes.append("dm_exgal < 0: NE2025 disk DM exceeds observed DM")

    out.update(
        dm_ne2025_calc=None if dm_mw is None else round(dm_mw, 6),
        dm_ne2025_2dec_calc=None if dm_mw is None else round(dm_mw, 2),
        dm_ne2025_err_calc=None if dm_mw_err is None else round(dm_mw_err, 6),
        dm_mw_origin=dm_mw_origin,
        dm_exgal_calc=None if dm_exgal is None else round(dm_exgal, 6),
        dm_exgal_err_pos_calc=None if dm_exgal_err_pos is None else round(dm_exgal_err_pos, 6),
        dm_exgal_err_neg_calc=None if dm_exgal_err_neg is None else round(dm_exgal_err_neg, 6),
        dm_mwhalo_calc=halo_dm,
        notes_calc="; ".join(notes),
    )

    return out


# ======================================================================
# Column resolution for the current CSV names
# ======================================================================
def _norm(header: Any) -> str:
    return re.sub(r"\s+", " ", str(header).strip().strip('"')).lower()


_ACCEPT = {
    "frb": ["frb", "frb name", "name", "burst"],
    "l_deg": ["l_deg", "l (deg)", "l deg", "gl"],
    "l_err": ["l_err", "l err", "l_err_deg"],
    "b_deg": ["b_deg", "b (deg)", "b deg", "gb"],
    "b_err": ["b_err", "b err", "b_err_deg"],
    "ra_hms": ["ra_hms", "ra (hms)", "ra hms", "ra"],
    "ra_deg": ["ra_deg", "ra (deg)", "ra deg"],
    "ra_err_sec": ["ra_err_sec", "ra err sec", "ra_error_sec"],
    "ra_err_arcsec": ["ra_err_arcsec", "ra err arcsec", "ra_error_arcsec"],
    "dec_dms": ["dec_dms", "dec (dms)", "dec dms", "dec"],
    "dec_deg": ["dec_deg", "dec (deg)", "dec deg"],
    "dec_err": ["dec_err", "dec err", "dec_err_arcsec", "dec err arcsec"],
    "dm_obs": ["dm_obs", "dm obs", "dm obs (pc cm^-3)", "dm"],
    "dm_err_pos": ["dm_err_pos", "dm obs err pos", "dm_obs_err_pos", "dm error +"],
    "dm_err_neg": ["dm_err_neg", "dm obs err neg", "dm_obs_err_neg", "dm error -"],
    "dm_ne2025_2dec": ["dm_ne2025_2dec", "dm ne2025 2dec", "dm_ne2025"],
    "dm_mwhalo": ["dm_mwhalo", "dm mw halo", "dm_halo"],
}


def resolve_columns(header: list[str]) -> dict[str, Optional[int]]:
    normalized = [_norm(name) for name in header]
    result: dict[str, Optional[int]] = {}
    for logical_name, accepted_names in _ACCEPT.items():
        result[logical_name] = next(
            (index for index, name in enumerate(normalized) if name in accepted_names),
            None,
        )
    return result


# ======================================================================
# CSV processing
# ======================================================================
def process_csv(input_csv: str = INPUT_CSV, output_csv: Optional[str] = OUTPUT_CSV) -> pd.DataFrame:
    if not _HAVE_ASTROPY:
        print("WARNING: astropy not found; coordinate-frame and l/b error conversion may be skipped.")
    if not _HAVE_MWPROP:
        print("WARNING: mwprop not found; NE2025 DM will be skipped unless dm_ne2025_2dec is populated.")

    raw = pd.read_csv(input_csv, header=0, dtype=str, keep_default_na=False)
    header = list(raw.columns)
    column_index = resolve_columns(header)

    frb_index = column_index.get("frb")
    if frb_index is None:
        raise ValueError(f"No FRB-name column found. Headers were: {header}")

    def cell(row_number: int, logical_name: str):
        index = column_index.get(logical_name)
        return None if index is None else raw.iat[row_number, index]

    keep_rows = [
        row_number
        for row_number in range(len(raw))
        if clean(raw.iat[row_number, frb_index]) is not None
    ]

    derived_records: list[dict[str, Any]] = []
    for row_number in keep_rows:
        logical_row = {
            "l_deg": cell(row_number, "l_deg"),
            "l_err": cell(row_number, "l_err"),
            "b_deg": cell(row_number, "b_deg"),
            "b_err": cell(row_number, "b_err"),
            "ra_hms": cell(row_number, "ra_hms"),
            "ra_deg": cell(row_number, "ra_deg"),
            "ra_err_sec": cell(row_number, "ra_err_sec"),
            "ra_err_arcsec": cell(row_number, "ra_err_arcsec"),
            "dec_dms": cell(row_number, "dec_dms"),
            "dec_deg": cell(row_number, "dec_deg"),
            "dec_err": cell(row_number, "dec_err"),
            "dm_obs": cell(row_number, "dm_obs"),
            "dm_err_pos": cell(row_number, "dm_err_pos"),
            "dm_err_neg": cell(row_number, "dm_err_neg"),
            "dm_ne2025_2dec": cell(row_number, "dm_ne2025_2dec"),
            "dm_mwhalo": cell(row_number, "dm_mwhalo"),
        }
        derived_records.append(derive_burst(logical_row))

    original = raw.loc[keep_rows].reset_index(drop=True)
    derived = pd.DataFrame(derived_records).reset_index(drop=True)
    result = pd.concat([original, derived], axis=1)

    if output_csv is None:
        stem, _ = os.path.splitext(input_csv)
        output_csv = f"{stem}_processed.csv"

    result.to_csv(output_csv, index=False)
    print(f"Wrote {len(result)} bursts -> {output_csv}")
    print(f"  coordinates available for {derived['ra_deg_calc'].notna().sum()} bursts")
    print(f"  dm_exgal computed for {derived['dm_exgal_calc'].notna().sum()} bursts")
    return result


# ======================================================================
# Manual entry for one burst
# ======================================================================
def manual_burst(**fields: Any) -> dict[str, Any]:
    """Calculate derived fields for one burst using the current logical names.

    Example:
        manual_burst(
            ra_hms="18h50m40.8s",
            dec_dms="+70d14m37.8s",
            ra_err_arcsec=2.19,
            dec_err=1.49,
            dm_obs=269.53,
            dm_err_pos=0.1,
            dm_err_neg=0.1,
            dm_mwhalo=30,
        )
    """
    return derive_burst(fields)


if __name__ == "__main__":
    process_csv()
