import mwprop
from mwprop.nemod.NE2025 import ne2025

frb_names = [
    "20121102A",
    "20180916B",
    "20180924A",
    "20181112A",
    "20190102B"
]

l_list = [
    174.9,
    129.7,
    0.74,
    342.6,
    312.6
]

b_list = [
    -0.2,
    3.7,
    -49.4,
    -47.7,
    -33.5
]

dm_list = [
    557,
    349,
    361,
    589,
    364
]


frb_names.extend([
    "20190523A", "20190608B", "20190611B", "20190711A",
    "20190714A", "20191001A", "20200430A", "20200120E",
    "20201124A"
])

l_list.extend([
    117.0, 53.2, -47.1, -49.1, -71.1,
    -17.3, 17.1, 142.2, 177.8
])

b_list.extend([
    44.0, -48.5, -33.3, -33.9, 48.7,
    -44.0, 52.5, 41.22, -8.52
])

dm_list.extend([
    761, 339, 321, 593, 504,
    507, 380, 88, 414
])

cutoffs = [0.001, 0.01, 0.1, 1, 10, 100] # in kpc

## This code is for testing the convergence of the NE2025 model's DM predictions as a function of integration distance.
def test_dm_convergence():
    import matplotlib.pyplot as plt
    import numpy as np

    dmin = 10**-3
    dmax = 100
    nd = 100
    cutoffs = 10**np.linspace(np.log10(dmin), np.log10(dmax), nd)

    results = {}

    for cutoff in cutoffs:
        results[cutoff] = []

        for name, l, b in zip(frb_names, l_list, b_list):

            Dk, Dv, Du, Dd = ne2025(
                ldeg=l,
                bdeg=b,
                dmd=cutoff,
                ndir=-1,
                classic=False
            )

            print(f"{name}: {cutoff:.4g} kpc, DM: {Dv['DM']} pc cm^-3")
            results[cutoff].append(Dv["DM"])

    for i, name in enumerate(frb_names):
        y = [results[c][i] for c in cutoffs]
        plt.plot(cutoffs, y, marker="o", markersize=4, label=name)

    plt.xscale("log")
    plt.xlabel("Integration Distance (kpc)")
    plt.ylabel("Predicted Galactic DM (pc cm$^{-3}$)")
    plt.title("NE2025 DM Convergence")
    plt.legend()
    plt.show()

    return results

## This code plots the predicted Galactic DM from the NE2025 model as a function of absolute Galactic latitude for the FRBs in our sample.
def plot_latitude_vs_dm():

    import matplotlib.pyplot as plt
    import numpy as np

    dm_mw = []

    for l, b in zip(l_list, b_list):

        Dk, Dv, Du, Dd = ne2025(
            ldeg=l,
            bdeg=b,
            dmd=50,
            ndir=-1,
            classic=False
        )

        dm_mw.append(Dv["DM"])

    abs_b = np.abs(b_list)

    plt.figure(figsize=(8, 5))

    plt.scatter(abs_b, dm_mw, s=100)

    for x, y, label in zip(abs_b, dm_mw, frb_names):

        if label == "20180924A":
            offset = (8, 5)

        elif label == "20181112A":
            offset = (-60, 7)

        else:
            offset = (6, 4)

        plt.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points"
        )

    plt.xlabel(r"Absolute Galactic Latitude $|b|$ (deg)")
    plt.ylabel(r"Galactic DM (pc cm$^{-3}$)")
    plt.title("NE2025 Galactic DM vs Galactic Latitude")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## This code plots the predicted Galactic DM from the NE2025 model as a function of absolute Galactic longtitude for the FRBs in our sample.
def plot_longtitude_vs_dm():

    import matplotlib.pyplot as plt
    import numpy as np

    dm_mw = []

    for l, b in zip(l_list, b_list):

        Dk, Dv, Du, Dd = ne2025(
            ldeg=l,
            bdeg=b,
            dmd=50,
            ndir=-1,
            classic=False
        )

        dm_mw.append(Dv["DM"])

    abs_l = np.abs(l_list)

    plt.figure(figsize=(8, 5))

    plt.scatter(abs_l, dm_mw, s=100)

    for x, y, label in zip(abs_l, dm_mw, frb_names):

        if label == "20180924A":
            offset = (8, 5)

        elif label == "20181112A":
            offset = (-60, 7)

        else:
            offset = (6, 4)

        plt.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points"
        )

    plt.xlabel(r"Absolute Galactic Longitude $|l|$ (deg)")
    plt.ylabel(r"Galactic DM (pc cm$^{-3}$)")
    plt.title("NE2025 Galactic DM vs Galactic Longitude")
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## This code plots the total observed DM of each FRB against the predicted Galactic DM from the NE2025 model.
def plot_total_dm_vs_galactic_dm():
    import matplotlib.pyplot as plt

    dm_mw = []

    for l, b in zip(l_list, b_list):
        Dk, Dv, Du, Dd = ne2025(
            ldeg=l,
            bdeg=b,
            dmd=30,
            ndir=-1,
            classic=False
        )
        dm_mw.append(Dv["DM"])

    plt.figure(figsize=(8, 5))
    plt.scatter(dm_mw, dm_list, s=100)

    for x, y, label in zip(dm_mw, dm_list, frb_names):
        if label == "20180924A":
            offset = (-17, 12)

        elif label == "20190102B":
            offset = (12, 4)

        else:
            offset = (6, 4)

        plt.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points"
        )


    plt.xlabel(r"NE2025 Galactic DM (pc cm$^{-3}$)")
    plt.ylabel(r"Total Observed DM (pc cm$^{-3}$)")
    plt.title("Observed DM vs NE2025 Galactic DM")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

## This code plots the DM excess for each FRB after subtracting the predicted Galactic DM from the NE2025 model.
def plot_dm_excess():
    import matplotlib.pyplot as plt

    dm_mw = []

    for l, b in zip(l_list, b_list):
        Dk, Dv, Du, Dd = ne2025(
            ldeg=l,
            bdeg=b,
            dmd=30,
            ndir=-1,
            classic=False
        )
        dm_mw.append(Dv["DM"])

    dm_excess = [total - mw for total, mw in zip(dm_list, dm_mw)]

    plt.figure(figsize=(9, 5))
    plt.bar(frb_names, dm_excess)

    plt.xlabel("FRB")
    plt.ylabel(r"DM Excess (pc cm$^{-3}$)")
    plt.title("FRB DM Excess After Subtracting NE2025 Galactic DM")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

## This code plots the fraction of the observed DM that can be attributed to the Milky Way according to the NE2025 model for each FRB.
def plot_mw_dm_fraction():
    import matplotlib.pyplot as plt

    dm_mw = []

    for l, b in zip(l_list, b_list):
        Dk, Dv, Du, Dd = ne2025(
            ldeg=l,
            bdeg=b,
            dmd=30,
            ndir=-1,
            classic=False
        )
        dm_mw.append(Dv["DM"])

    fractions = [mw / total for mw, total in zip(dm_mw, dm_list)]

    plt.figure(figsize=(9, 5))
    plt.bar(frb_names, fractions)

    plt.xlabel("FRB")
    plt.ylabel(r"$DM_{MW} / DM_{total}$")
    plt.title("Fraction of Observed DM Accounted for by NE2025 Milky Way DM")
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Function calls
# test_dm_convergence()