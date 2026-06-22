from astropy.coordinates import SkyCoord
import astropy.units as u

# # Fill this list with your coordinates in decimal degrees.
# # Example format:
# # ra_dec_list = [
# #     {"name": "source_1", "ra_deg": 10.274058, "dec_deg": 21.226270},
# #     {"name": "source_2", "ra_deg": 123.456789, "dec_deg": -12.345678},
# # ]
# ra_dec_list = [
#     {"name": "placeholder_1", "ra_deg": 10.274058, "dec_deg": 21.226270},
# ]

# for source in ra_dec_list:
#     coord = SkyCoord(
#         ra=source["ra_deg"] * u.deg,
#         dec=source["dec_deg"] * u.deg,
#         frame="icrs",
#     )
#     galactic = coord.galactic

#     print(source["name"])
#     print(f"  RA:  {coord.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)}")
#     print(f"  Dec: {coord.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True, pad=True)}")
#     print(f"  l:   {galactic.l.deg:.6f} deg")
#     print(f"  b:   {galactic.b.deg:.6f} deg")


c = SkyCoord(
    ra=151.66159*u.deg,
    dec=35.69410*u.deg,
    frame='icrs'
)

print(c.galactic.l.deg)
print(c.galactic.b.deg)