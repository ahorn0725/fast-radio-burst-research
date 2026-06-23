from astropy.coordinates import SkyCoord
import astropy.units as u


# --------------------------------------------------
# 1. RA/Dec (hms, dms) -> Galactic (l, b)
# --------------------------------------------------
def radec_to_lb(ra_hms, dec_dms):
    """
    Parameters
    ----------
    ra_hms : str
        Example: '05h31m58.70s'
    dec_dms : str
        Example: '+33d08m52.5s'

    Returns
    -------
    l_deg, b_deg : float
        Galactic longitude and latitude in degrees
    """

    coord = SkyCoord(ra=ra_hms, dec=dec_dms, frame='icrs')
    gal = coord.galactic

    return gal.l.deg, gal.b.deg


# --------------------------------------------------
# 2. Galactic (l, b) -> RA/Dec
# --------------------------------------------------
def lb_to_radec(l_deg, b_deg):
    """
    Returns RA/Dec both as decimal degrees and HMS/DMS strings.
    """

    gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame='galactic'
    )

    icrs = gal.icrs

    ra_hms = icrs.ra.to_string(unit=u.hour,
                               sep=':',
                               precision=3)

    dec_dms = icrs.dec.to_string(
        unit=u.deg,
        sep=':',
        precision=3,
        alwayssign=True
    )

    return {
        "ra_deg": icrs.ra.deg,
        "dec_deg": icrs.dec.deg,
        "ra_hms": ra_hms,
        "dec_dms": dec_dms
    }


# --------------------------------------------------
# 3. RA uncertainty (seconds of time) -> arcseconds
# --------------------------------------------------
def ra_error_sec_to_arcsec(ra_error_sec, dec_deg=0):
    """
    Converts an RA error given in seconds of time
    into angular arcseconds.

    Uses:
        1 second RA = 15 arcsec * cos(dec)

    Parameters
    ----------
    ra_error_sec : float
        RA uncertainty in seconds of time

    dec_deg : float
        Declination in degrees

    Returns
    -------
    float
        Angular uncertainty in arcseconds
    """

    import numpy as np

    return ra_error_sec * 15 * np.cos(np.radians(dec_deg))


# --------------------------------------------------
# 4. RA/Dec decimal degrees -> HMS/DMS
# --------------------------------------------------
def deg_to_hmsdms(ra_deg, dec_deg):
    """
    Convert decimal degree coordinates
    into HMS/DMS strings.
    """

    coord = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        frame='icrs'
    )

    ra_hms = coord.ra.to_string(
        unit=u.hour,
        sep=':',
        precision=3
    )

    dec_dms = coord.dec.to_string(
        unit=u.deg,
        sep=':',
        precision=3,
        alwayssign=True
    )

    return ra_hms, dec_dms


# AUTOMATION IDEA
# take in some inputs and output every single value necessary based on the input values? like i can have the option to input every single
# type of input, but if i put like "x" or something then it'll skip and calculate it for me so it's easier to automate. also potentially
# could just put in any form of the dms or hms format and make it so that it is reformatted so i can use astropy with it or something. 
# e.g. i have ra/dec, ra/dec err, z, and i want to calc l/b and l/b error as well as dm. the ra/dec is like 4h40m38.2s or something and dec is
# -53d36m00s, and it can just reformat it with the colons so it'll be easier to calculate. 

