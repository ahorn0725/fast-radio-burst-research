"""
plot_host_dm_tau.py

Host-galaxy DM and cloudlet-model scattering vs. assumed host redshift
(Cordes, Ocker & Chatterjee 2022, Fig. 9), driven by the Nexus catalogue CSV.

Two equations do all the work:

    DM_h(z)   = (1 + z) * [DM_exgal - DM_halo - DM_IGM(z)]        (source frame)
    tau_h(z)  = 0.48 ms * A_tau*FGtilde * (DM_h/100)^2 * nu^-4 / (1 + z)^3

DM_IGM is the mean Macquart relation; the cyan band is its cosmic variance
carried through into DM_h and then into tau.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG = {
    "catalogue": "frb_catalogue.csv",
    "sep": ",",

    # Which FRBs to plot, by the value in the name column. One column of
    # panels per entry.
    "frbs": ["20121102A", "20190523A"],

    # Map plotter role -> column name in your catalogue.
    "columns": {
        "name": "frb_name",
        "dm": "dm_exgal",        # see dm_is_exgal below
        "dm_mw": "dm_mw",        # only used if dm_is_exgal is False
        "z": "z",
        "tau": "tau_ms",         # measured/limit scattering time, optional
        "tau_freq": "tau_freq_ghz",   # frequency it was measured at, GHz
    },

    # True  -> "dm" column is DM_obs with the Galactic *disk* already removed
    #          (your NE2025 dm_exgal); the script subtracts DM_HALO only.
    # False -> "dm" is total DM_obs; the script subtracts dm_mw AND DM_HALO.
    "dm_is_exgal": True,
    "dm_halo": 52.5,

    # Cosmology and IGM.
    "h": 0.677,
    "omega_m": 0.310,
    "omega_b": 0.0490,
    "f_e": 0.88,          # free electrons per baryon mass, fully ionised
    "f_igm": 0.85,

    # Cloudlet model. The paper's in-panel annotation is [0.01, 10]; the thick
    # line is the geometric mean of the two.
    "a_tau_fg": (0.01, 10.0),
    "nu_ghz": 1.0,        # frequency the panels are drawn at

    # Cosmic variance of DM_IGM: sigma/<DM> = cv_c / sqrt(z), capped at cv_cap.
    # Reproduces the published band; replace if your group uses another form.
    "cv_c": 0.20,
    "cv_cap": 0.95,

    "z_pad": 1.4,         # x-axis runs to z_pad * (z where DM_h -> 0)
    "outfile": "host_dm_tau.png",
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def dm_igm(z, cfg):
    """Mean Macquart-relation DM_IGM(z), pc cm^-3. Vectorised over z."""
    # n_e0 = f_igm * f_e * Omega_b * rho_crit / m_p   [cm^-3]
    rho_crit = 1.87834e-29 * cfg["h"] ** 2                  # g cm^-3
    n_e0 = cfg["f_igm"] * cfg["f_e"] * cfg["omega_b"] * rho_crit / 1.67262192e-24
    d_h = 2.99792458e5 / (100.0 * cfg["h"]) * 1.0e6          # c/H0 in pc

    zg = np.linspace(0.0, max(np.max(z), 1e-6), 4000)
    e_z = np.sqrt(cfg["omega_m"] * (1 + zg) ** 3 + 1 - cfg["omega_m"])
    integral = np.concatenate([[0.0], np.cumsum(np.diff(zg) *
                                                0.5 * ((1 + zg[1:]) / e_z[1:] +
                                                       (1 + zg[:-1]) / e_z[:-1]))])
    return n_e0 * d_h * np.interp(z, zg, integral)


def cv_frac(z, cfg):
    """Fractional 1-sigma cosmic variance of DM_IGM."""
    z = np.asarray(z, float)
    frac = np.full_like(z, cfg["cv_cap"])
    good = z > 0
    frac[good] = cfg["cv_c"] / np.sqrt(z[good])
    return np.clip(frac, 0.0, cfg["cv_cap"])


def dm_host(z, dm_ex_minus_halo, cfg, igm_scale=1.0):
    """DM_h in the host rest frame, pc cm^-3. NaN where the budget runs out."""
    dm_h = (1 + z) * (dm_ex_minus_halo - igm_scale * dm_igm(z, cfg))
    return np.where(dm_h > 0, dm_h, np.nan)


def tau_host(dm_h, z, a_tau_fg, nu_ghz):
    """Observer-frame host scattering time, ms."""
    return 0.48 * a_tau_fg * (dm_h / 100.0) ** 2 * nu_ghz ** -4 / (1 + z) ** 3


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def panel(ax_dm, ax_tau, row, cfg):
    dm_budget = row["dm_budget"]      # DM_exgal - DM_halo
    z_meas = row["z"]

    # Where the mean curve hits zero sets a sensible x-range.
    ztest = np.linspace(0, 3, 3000)
    zero = ztest[np.argmin(np.abs(dm_budget - dm_igm(ztest, cfg)))]
    z = np.linspace(0.0, cfg["z_pad"] * zero, 800)

    s = cv_frac(z, cfg)
    dm_mid = dm_host(z, dm_budget, cfg, 1.0)
    dm_hi = dm_host(z, dm_budget, cfg, 1 - s)    # low DM_IGM -> high DM_h
    dm_lo = dm_host(z, dm_budget, cfg, 1 + s)

    fg_lo, fg_hi = cfg["a_tau_fg"]
    fg_mid = np.sqrt(fg_lo * fg_hi)
    nu = cfg["nu_ghz"]

    for ax, mid, lo, hi in (
        (ax_dm, dm_mid, dm_lo, dm_hi),
        (ax_tau,
         tau_host(dm_mid, z, fg_mid, nu),
         tau_host(dm_lo, z, fg_lo, nu),
         tau_host(dm_hi, z, fg_hi, nu)),
    ):
        ax.fill_between(z, np.nan_to_num(lo), np.nan_to_num(hi),
                        color="#bdf0f7", lw=0)
        ax.plot(z, lo, "k", lw=0.8)
        ax.plot(z, hi, "k", lw=0.8)
        ax.plot(z, mid, "k", lw=2.5)
        ax.axvline(z_meas, color="#e8402a", lw=2)
        ax.set_yscale("log")
        ax.set_xlim(0, z[-1])
        ax.tick_params(direction="in", top=True, right=True, which="both")

    # FGtilde-only band, drawn on top of the cosmic-variance band.
    ax_tau.fill_between(z,
                        np.nan_to_num(tau_host(dm_mid, z, fg_lo, nu)),
                        np.nan_to_num(tau_host(dm_mid, z, fg_hi, nu)),
                        color="#3ec8dc", lw=0, zorder=1.5)
    ax_tau.plot(z, tau_host(dm_mid, z, fg_mid, nu), "k", lw=2.5, zorder=3)

    if np.isfinite(row.get("tau_1ghz", np.nan)):
        ax_tau.axhline(row["tau_1ghz"], color="#2e7d32", lw=1.8, ls="--")

    ax_dm.set_ylim(5, 1.5e3)
    ax_tau.set_ylim(1e-3, 2e3)
    ax_dm.set_ylabel(r"$\widehat{\mathrm{DM}}_\mathrm{h}$  (pc cm$^{-3}$)")
    ax_tau.set_ylabel(rf"$\widehat{{\tau}}_\mathrm{{h}}$  (ms, {nu:g} GHz obs)")
    ax_tau.set_xlabel(r"Host galaxy redshift  $z_\mathrm{h}$")
    ax_dm.tick_params(labelbottom=False)
    ax_dm.set_title(f"{row['name']}   "
                    rf"DM$_\mathrm{{budget}}$ = {dm_budget:.0f},  "
                    rf"$f_\mathrm{{igm}}$ = {cfg['f_igm']},  "
                    rf"$A_\tau\widetilde{{F}}G$ = [{fg_lo:g}, {fg_hi:g}]",
                    fontsize=10)


def load(cfg):
    """Pull the rows we need out of the catalogue and normalise them."""
    col = cfg["columns"]
    df = pd.read_csv(cfg["catalogue"], sep=cfg["sep"])
    rows = []
    for frb in cfg["frbs"]:
        r = df[df[col["name"]].astype(str).str.contains(frb, na=False)]
        if r.empty:
            raise SystemExit(f"{frb} not found in {cfg['catalogue']}")
        r = r.iloc[0]

        dm = float(r[col["dm"]])
        if not cfg["dm_is_exgal"]:
            dm -= float(r[col["dm_mw"]])
        dm -= cfg["dm_halo"]

        tau_1ghz = np.nan
        if col.get("tau") in df.columns and np.isfinite(pd.to_numeric(
                r.get(col["tau"]), errors="coerce")):
            nu_obs = float(r.get(col["tau_freq"], 1.0))
            # scale the measured tau to the plotted frequency: tau ~ nu^-4
            tau_1ghz = float(r[col["tau"]]) * (nu_obs / cfg["nu_ghz"]) ** 4

        rows.append({"name": frb, "dm_budget": dm,
                     "z": float(r[col["z"]]), "tau_1ghz": tau_1ghz})
    return rows


def main():
    cfg = CONFIG
    rows = load(cfg)
    n = len(rows)
    fig, axs = plt.subplots(2, n, figsize=(6.5 * n, 7),
                            squeeze=False, gridspec_kw=dict(hspace=0.05))
    for i, row in enumerate(rows):
        panel(axs[0, i], axs[1, i], row, cfg)
        print(f"{row['name']}: DM_budget = {row['dm_budget']:.1f}, "
              f"z = {row['z']:.3f}, "
              f"DM_IGM = {dm_igm(np.array([row['z']]), cfg)[0]:.1f}")
    fig.savefig(cfg["outfile"], dpi=160, bbox_inches="tight")
    print("wrote", cfg["outfile"])


if __name__ == "__main__":
    main()