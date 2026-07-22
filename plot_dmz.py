"""
SC 20201130: Plot DM-z relationship
Use table known-redshifts.txt:
# frb_name  gl, gb  DM, dm_err mw_dm_limit NEmodel  z_host  Rep?  Grade

SC 20221114: Updated to include FRB at z=1 and other entries; extend DM-z
SC 20210601: Also plot cosmic variance band for DM_IGM
SC 20210913: Add label for FRB 121102; identify repeaters
SC 20210918: Change f_IGM to 0.85 and const = 978 pc/cc
SC 20211128: Change to read in dm-z from JMC's DMigm.py; halo = 25-80 dmu.
	Now IGM band is in IGM_z.txt
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt


params = {"figure.figsize": (12,8),
			"pdf.fonttype": 42,
			"ps.fonttype":42,
             "font.size": 12,
             "font.family": 'Helvetica',
             "font.weight": "normal"}
matplotlib.rcParams.update(params)
# Note, forcing the use of Type 4.2 fonts, rather than Type 3.


#from matplotlib import gridspec

# Load in arrays
data = pd.read_csv('dm_z_graphing.tsv', sep="\t")
z = data['z']
dmg = data['dm_ne2025_2dec']
dmh    = 52.5  # Was 50 +- 25
dmherr = 27.5  # Now reflects range of 25--80.
dmx = data['dm_obs'] - data['dm_ne2025_2dec'] - dmh
dmxerr = dmherr + 0.4*data['dm_ne2025_2dec']
repeats = data['repeater']
# repeater = np.where(repeats)
# oneoff   = np.where(repeats==0)

# Colors
psrf = '#80d4ff'
lmcf = '#ffff66'
smce = '#e68a00'
smcf = '#ffff66'
cne2k = '#009933'
chalo = '#006699'
cfrb  = '#aa0033'
cgray = '#888888'
cxsdm = '#e60045'
czdm1 = '#e63900'
czdm2 = 'orange'


"""
DM-z lines now come from IGM_z.txt
simdm = np.arange(40,690,10)
simz = simdm/1000.0
# Also note, from Shull & Danforth 2018, f_IGM = 0.6; we choose f_IGM=0.8 -> 0.85
# DM_IGM = 978 pc/cc * f_IGM * z
# sigma DM = sqrt(50*DM_IGM)
# Note, I need to go from DM(z) to z(DM) - may be easiest to do this with polygon fill
fIGM = 0.85
DMconst = 978
zdm = simdm / (fIGM*DMconst)
valid = np.where(zdm<0.7) # Where this plot needs to top out
## Generate the points at the corner of the polygon
zlow = zdm[valid][0]; zhigh = zdm[valid][-1]
dmlow = simdm[valid][0]; dmhigh = simdm[valid][-1]
sdmlow = np.sqrt(dmlow * 50.0)
sdmhigh = np.sqrt(dmhigh * 50.0)
fill_dm = [max(0,dmlow-sdmlow), dmlow+sdmlow, dmhigh+sdmhigh, dmhigh-sdmhigh]
fill_z = [zlow, zlow, zhigh, zhigh]
## Pass to plt.fill
"""

# Plot
plt.errorbar(-(dmg), z, None, (np.zeros(len(dmg)),dmg), color=cne2k, marker='o', ls='', label='MW disk')
#plt.plot(-(dmg+dmh), z, ls='', color=chalo, marker='o', markersize=3)
plt.errorbar(-(dmg+dmh), z, None, dmherr, capsize=5, marker=None, markersize=3, ls='', ecolor=chalo, label='MW halo')
#plt.errorbar(dmx, z, None, dmherr, color=cfrb, marker='s', ls='', label='Extragalactic')

plt.errorbar(dmx[repeats==0], z[repeats==0], None, dmxerr[repeats==0], \
	color=cfrb, marker='o', mfc='white', ls='', label='One-offs')
plt.errorbar(dmx[repeats==1], z[repeats==1], None, dmxerr[repeats==1], \
	color=cfrb, marker='s', ls='', label='Repeaters')

# Simple version: DM = 1000.z
# plt.plot(simdm,simz, ls='-', lw=0.7, color=czdm1, label='DM-z v1')
# plt.plot(simdm[valid], zdm[valid], ls='-', lw=0.7, color=czdm2, label='DM-z')
# plt.fill(fill_dm, fill_z, color=czdm2, alpha=0.15)

(simz, dmmed, dmmin, dmmax) = np.loadtxt('igm_z_extended.txt', unpack=True)
plt.plot(dmmed, simz, ls='-', lw=0.7, color=czdm2, label='DM-z')
plt.fill_betweenx(simz, dmmin, dmmax, color=czdm2, alpha=0.15)

plt.legend(loc=4, title='n = 126 FRBs')
# plt.xlabel(r'($\leftarrow$ Galactic | Extragalactic $\rightarrow$)    Dispersion Measure (pc cm$^{-3}$)')
plt.xlabel(r'Dispersion Measure (pc cm$^{-3}$)')
plt.ylabel(r'Host Redshift $z$')

val = np.arange(-300, 1501, 150)
txt = np.abs(val)
plt.xticks(val, txt)

# Add axis separator
plt.plot([0,0.05],[0,1.4],ls=':', color=cgray)

# Add redshift ticks and labels on both sides
ax = plt.gca()
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelright=True)

# Add a line to draw attention to FRB20190520 DONE
plt.text(1200,0.255, 'FRB20190520B', horizontalalignment='center', color=cxsdm)

# Add a label for FRB20121102 DONEISH?
plt.text(275,0.17, 'FRB20121102', va='center_baseline', ha='left', color=cxsdm)

# Add a label for FRB20220610A DONE
plt.text(1480,1.05, 'FRB20220610A', va='center_baseline', ha='right', color=cxsdm)

# Add a label for FRB20230521B DONE
plt.text(1180,1.32, 'FRB20230521B', va='center_baseline', ha='right', color=cxsdm)


plt.annotate(r'$\leftarrow$ Galactic | Extragalactic $\rightarrow$',(0.2047,0.01), xycoords='axes fraction')
plt.annotate(r'$\leftarrow$ Galactic | Extragalactic $\rightarrow$',(0.2047,0.96), xycoords='axes fraction')
# plt.annotate(r'$\leftarrow$ Galactic | Extragalactic $\rightarrow$',(108,-45), xycoords='axes pixels')

plt.subplots_adjust(left=0.07, right=0.95, top=0.96, bottom=0.09)
plt.show()
# plt.savefig('fig_dmz.pdf')
