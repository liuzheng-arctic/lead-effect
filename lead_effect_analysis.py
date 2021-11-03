from pathlib import Path
import numpy as np
import h5py 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from  cartopy.feature import OCEAN as cOCEAN

from leads.grid import Grid_Coarse as CGrid
from leads.gridgran import Gridded_Granule as ggran
from leads.database import is2db
from leads.utils import mapset,ptproj


# --- SET UP map and grid
mds,mapproj = mapset('LAEA',WIDTH=6600e3,dx=200e3,dy=200e3)
pcproj = ccrs.PlateCarree()
DX = mds.dx
DY = mds.dy
XLIM = mds.XLIM
YLIM = mds.YLIM
NY = int( YLIM/DY*2 )
NX = int( XLIM/DX*2 )
xgrid = np.linspace(-XLIM,XLIM,NX+1)
ygrid = np.linspace(-YLIM,YLIM,NY+1)


slat = mds.lat.values
slon = mds.lon.values
sxx,syy = ptproj(pcproj,mapproj,slon,slat)
xb = np.linspace(-XLIM,XLIM,NX+1)
yb = np.linspace(-YLIM,YLIM,NY+1)
xbs,ybs = np.meshgrid(xb,yb)

# --- SET UP path to monthly aggregated data
path0 = Path('/media/siz/wrf/IS2/mon_LAEA_granule_subgrid/V004/200x200/ATL07/1')
db_grid = CGrid.grid_db(path0)
tgrd = CGrid(grid_root=path0)

# --- SET UP plotting settings
ftsz = 16
lw   = 2
import matplotlib
matplotlib.rc('xtick', labelsize=ftsz-2)
matplotlib.rc('ytick', labelsize=ftsz-2)
matplotlib.rc('font',**{'family':'serif','variant':'small-caps'}) 


# --- SETTING for plot saving
picdir = Path.home()/'Outpics/IDS_LEAD/20211029'
picfmt = 'png'
picdpi = 300
if not picdir.exists(): picdir.mkdir(parents=True, exist_ok=True)


# ### Define a convenient plot function
def plot_map(
    x,y,dat,fig, ax, mapproj,DX,DY,
    #vmin=.60,vmax=1.00,
    vmin=None,vmax=None,
    nclrs=20,nctick=5,ttl_txt=None,ftsz=18,
    cmap=plt.cm.RdBu_r,
    pixel=False,
    ):
    mxx = x*1.
    myy = y*1.
    mdat = dat
    xmin = mxx.min()
    xmax = mxx.max()
    ymin = myy.min()
    ymax = myy.max()
    if pixel:
        xmin += DX/2
        xmax -= DX/2
        ymin += DY/2
        ymax -= DY/2
        
        mxx -= DX/2
        myy -= DX/2
    if vmin is None: vmin = np.nanmin(mdat)
    if vmax is None: vmax = np.nanmax(mdat)
        
    clvl = np.linspace(vmin,vmax,nclrs+1)
    mdat[mdat==0] = np.nan
    ax.set_extent([xmin,xmax,ymin,ymax],crs=mapproj)
    ax.coastlines(color='0.3',resolution='110m',linewidth=2)
    ax.gridlines(xlocs=np.arange(-180,180,45))
    if not pixel:
        chdl = ax.contourf(mxx,myy,mdat,cmap=cmap,
                        vmin=vmin,vmax=vmax,levels=clvl,
                        extend='both')
    else: 
        chdl = ax.pcolormesh(mxx,myy,mdat,cmap=cmap,
                        vmin=vmin,vmax=vmax,
                        )
    if ttl_txt is not None:
        ax.set_title(ttl_txt,fontsize=ftsz)

    return chdl


# ======================================================
# --- seasonal cloud cover maps
# ======================================================
sea_mon = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
sea_txt = ['Winter','Spring','Summer','Autumn']
fig = plt.figure('all',figsize=(10,10))
for i in range(4):
    db_season = db_grid[db_grid.month.isin(sea_mon[i])]
    dat_season = CGrid(grid_db=db_season)
    mseason= dat_season.mean(dat_season.data,dim='time')
    dat = mseason['cld_low_all']#.values
    dat_nz = dat.where(mseason['nprof_floe']>0,drop=True)
    print(sea_txt[i],dat_nz.mean().values,dat.mean().values)
    ax = fig.add_subplot(2,2,i+1,projection=mapproj)
    chdl = plot_map(
        sxx,syy,dat.values,fig,ax,mapproj,DX,DY,
        vmin=0.,
        lowres=False,ttl_txt=sea_txt[i],cmap=plt.cm.RdBu_r)
fig.subplots_adjust(hspace=.1,wspace=0,left=.01,right=.87,top=.95,bottom=.05)
cb_ax = fig.add_axes([0.89, 0.05, 0.05, 0.9])
cbar = fig.colorbar(chdl, cax=cb_ax,ticks=np.linspace(.0,1.00,5))
cbar.ax.tick_params(labelsize=16)
picfn = 'all_cld_season.{0}'.format(picfmt)
picfn = picdir/picfn
plt.savefig(picfn,format=picfmt,dpi=picdpi)



# ======================================================
# --- lead effect on cloud plume fraction
# ======================================================

NPROF_MIN = 100
mons = [12,1,2,3,4]

i_DLEAD = 2
i_plume = 1
cbfrac = .075

# -----
db_season = db_grid[db_grid.month.isin(mons)]
dat_season = CGrid(grid_db=db_season)
mseason= dat_season.mean(dat_season.data,dim='time')

np_floe = mseason['nprof_floe']
np_floe_norm = np_floe/mseason['n_granule']

dat0 = mseason['cld_plume_lead'].isel(D_LEAD=i_DLEAD,ZBOT_PLUME=i_plume)
dat1 = mseason['cld_plume_floe'].isel(ZBOT_PLUME=i_plume)
ddat = mseason['cld_plume_diff'].isel(D_LEAD=i_DLEAD,ZBOT_PLUME=i_plume)
dat0 = dat0.where(np_floe_norm>NPROF_MIN)
dat1 = dat1.where(np_floe_norm>NPROF_MIN)
ddat = ddat.where(np_floe_norm>NPROF_MIN)


pixel = False
# -----
fig = plt.figure('plume',figsize=(12,5))
ax = fig.add_subplot(1,3,1,projection=mapproj)
ax.add_feature(cOCEAN,color='.5')
chdl = plot_map(sxx,syy,dat0.values,fig,ax,mapproj,DX,DY,
                vmin=0,lowres=False,pixel=pixel)
cbar = fig.colorbar(chdl,orientation='horizontal', aspect=10,fraction=cbfrac,pad=.05,
                    ticks=np.linspace(0,1.00,5))
cbar.ax.tick_params(labelsize=16)
# -----
ax = fig.add_subplot(1,3,2,projection=mapproj)
ax.add_feature(cOCEAN,color='.5')
chdl = plot_map(sxx,syy,dat1.values,fig,ax,mapproj,DX,DY,
                vmin=0,lowres=False,pixel=pixel)
cbar = fig.colorbar(chdl,orientation='horizontal',aspect=10,fraction=cbfrac,pad=.05,
                    ticks=np.linspace(0,1.00,5))
cbar.ax.tick_params(labelsize=16)
# -----
ax = fig.add_subplot(1,3,3,projection=mapproj)
ax.add_feature(cOCEAN,color='.5')
vmin = -.15
vmax = .0
chdl = plot_map(sxx,syy,ddat.values,fig,ax,mapproj,DX,DY,nclrs=15,
                lowres=False,vmin=vmin,vmax=vmax,pixel=pixel)
cbar = fig.colorbar(chdl,orientation='horizontal',aspect=10,fraction=cbfrac,pad=.05,
                    ticks=np.linspace(vmin,vmax,4))
cbar.ax.tick_params(labelsize=16)

# ----
plt.tight_layout()
fig.subplots_adjust(left=.02,right=.98,hspace=.0,wspace=.15)
picfn = 'lead_effets_subgrid_nprof{0}.{1}'.format(NPROF_MIN,picfmt)
picfn = picdir/picfn
plt.savefig(picfn,format=picfmt,dpi=picdpi)



# ======================================================
# --- Lead effect distribution ---
# ======================================================
NPROF_MIN = 100

mons = [12,1,2,3,4]
db_season = db_grid[db_grid.month.isin(mons)]
dat_season = CGrid(grid_db=db_season)
mseason= dat_season.mean(dat_season.data,dim='time')

i_DLEAD = 2
i_plume = 1

np_floe = mseason['nprof_floe']
np_floe_norm = np_floe/mseason['n_granule']



dat0 = mseason['cld_plume_lead'].isel(D_LEAD=i_DLEAD,ZBOT_PLUME=i_plume)
dat1 = mseason['cld_plume_floe'].isel(ZBOT_PLUME=i_plume)
ddat = mseason['cld_plume_diff'].isel(D_LEAD=i_DLEAD,ZBOT_PLUME=i_plume)*1
dat0 = dat0.where(nprof_floe_norm>NPROF_MIN)
dat1 = dat1.where(nprof_floe_norm>NPROF_MIN)
vdif = dat0-dat1

ddat = ddat.where(nprof_floe_norm>NPROF_MIN)
vdif = vdif.where(nprof_floe_norm>NPROF_MIN)
# -----
#  Note: 
#  ddat: from cld_plume_diff
#  vdif: from cld_plume_lead - cld_plume_floe

bin_edge = np.linspace(-1,1,51)
binc = (bin_edge[1:]+bin_edge[:-1])/2

dvec_vdif = vdif.values.ravel()
dvec_vdif = dvec_vdif[~np.isnan(dvec_vdif)]
dvec_ddat = ddat.values.ravel()
dvec_ddat = dvec_ddat[~np.isnan(dvec_ddat)]

hist_vdif = np.histogram(dvec_vdif,bins=bin_edge)[0]
hist_ddat = np.histogram(dvec_ddat,bins=bin_edge)[0]
hist_vdif = hist_vdif/(hist_vdif.sum())
hist_ddat = hist_ddat/(hist_ddat.sum())

plt.plot(binc,hist_ddat,'-kx',linewidth=2)
plt.grid(True)
plt.xlim(-.5,.5)
plt.xlabel('Cloud plume fraction difference',fontsize=ftsz)
plt.ylabel('Frequency of occurrence',fontsize=ftsz)
plt.tight_layout()
picfn = 'lead_effets_hist_sub.{0}'.format(picfmt)
picfn = picdir/picfn
plt.savefig(picfn,format=picfmt,dpi=picdpi)

# ======================================================
# --- Plot vertical profiles
# ======================================================
vcrit = -.1
vcrit_pos = -.077
vcrit_neg = -.123
i_DLEAD = 2
i_plume = 1


np_floe = mseason['nprof_floe']
np_floe_norm = np_floe/mseason['n_granule']


# -----
vdat = vdif
cld_floe = mseason.cldfrc_floe
cld_lead = mseason.cldfrc_lead.isel(D_LEAD=i_DLEAD)
cld_diff_sub = mseason.cldfrc_diff.isel(D_LEAD=i_DLEAD)
cld_floe = cld_floe.where(np_floe_norm>NPROF_MIN)
cld_lead = cld_lead.where(np_floe_norm>NPROF_MIN)
cld_diff = cld_lead - cld_floe
cld_diff_sub = cld_diff_sub.where(np_floe_norm>NPROF_MIN)


mcld_floe = cld_floe.mean(dim=('Y','X'))
mcld_lead = cld_lead.mean(dim=('Y','X'))
mcld_diff_sub = cld_diff_sub.mean(dim=('Y','X'))
mcld_diff = cld_diff.mean(dim=('Y','X'))

cld_diff_pos_sub = cld_diff_sub.where(ddat>vcrit_pos)
cld_diff_neg_sub = cld_diff_sub.where(ddat<vcrit_neg)
mcld_diff_pos_sub = cld_diff_pos_sub.mean(dim=('Y','X'))
mcld_diff_neg_sub = cld_diff_neg_sub.mean(dim=('Y','X'))

plt.figure(1,(8,6))
plt.subplot(121)
plt.plot(
    mcld_floe,mseason.Z/1000,'-b',
    mcld_lead,mseason.Z/1000,'-g',
    mcld_diff_sub,mseason.Z/1000,'-r',
    lw=2)
plt.ylim(0,12)
plt.xlim(-.1,.3)
plt.grid(True)
plt.legend(['sea ice','lead','lead-sea ice',
           ],fontsize=ftsz)
plt.xlabel('cloud fraction',fontsize=ftsz)
plt.ylabel('altitude [km]',fontsize=ftsz)
plt.subplot(122)
plt.plot(
    mcld_diff_pos_sub-mcld_diff_sub,mseason.Z/1000,'-k',
    mcld_diff_neg_sub-mcld_diff_sub,mseason.Z/1000,'--k',
    lw=2)
plt.ylim(0,12)
#plt.xlim(-.03,.03)
plt.grid(True)
plt.legend([r'$1^{st} quartile$',r'$4^{th} quartile$'],loc=1,fontsize=ftsz)
plt.xlabel('cloud fraction difference',fontsize=ftsz)

plt.tight_layout()
fig.subplots_adjust(left=.02,right=.98,hspace=.0,wspace=.02)
picfn = 'lead_effets_profile.{0}'.format(picfmt)
picfn = picdir/picfn
plt.savefig(picfn,format=picfmt,dpi=picdpi)

