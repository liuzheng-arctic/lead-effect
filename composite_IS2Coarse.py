from pathlib import Path
import numpy as np
import h5py 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gc, timeit
from datetime import datetime
from mpi4py import MPI
import sys

from leads.gridgran import Gridded_Granule as ggran
from leads.gridgran_coarse import GG_Coarse as ggc
from leads.gridgran_subgrid import GG_Subgrid as ggs
from leads.granule import Granule
from leads.database import is2db
from leads.utils import mapset,ptproj

def time_interval(tt,i):
    t00 = pd.Timestamp(year=tt[0].year,month=tt[0].month,day=1)
    t_start = t00 + pd.DateOffset(months=i)
    t_end = t00 + pd.DateOffset(months=i+1)
    return t_start, t_end

rebuild_db = True
save_log = True
PRD = 'ATL07'
version = 'V004'
beam_num = 1
feat_res = '200x200'

composite_dir = Path('/wrf/IS2/mon_LAEA_granule_subgrid')/version/feat_res/PRD/str(beam_num)
gridroot = Path('/wrf/IS2/LAEA_granule')/version/feat_res#/PRD/str(beam_num)
coarse_fn = 'coarse_DLEAD_{0}_{1}_{2}_gt{3}.grid'.format(version,feat_res,PRD,str(beam_num))
grid_db_dir = Path('.grid')
logdir= Path('./log_composite')
log_fn_out = '.log'
log_fn_err = '.err'

# --- nc compression setting ---
fcomp = dict(zlib=True, complevel=5, dtype='float32')

# --- mpi setting ---
PROC_ROOT = 0
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

if not grid_db_dir.exists(): grid_db_dir.mkdir(parents=True,exist_ok=True)
if not composite_dir.exists(): composite_dir.mkdir(parents=True,exist_ok=True)
coarse_fn = grid_db_dir/coarse_fn

if rebuild_db:
    coarse_df = is2db.build_db(gridroot,prd=PRD,beam=beam_num,subset='',filetype='nc',strong=False)
    coarse_df.to_csv(coarse_fn,index=False)
else:
    coarse_df = pd.read_csv(coarse_fn,parse_dates=['time'])
#sys.exit('here')    
    
tt = pd.to_datetime(coarse_df.time.values)
n_mon = (tt[-1].month-tt[0].month+1) + 12*(tt[-1].year-tt[0].year)

N0 = 0; N1 = n_mon
all_list = range(N0,N1)
mpi_idx = list(range(RANK,len(all_list),NPROC))
mon_idx = np.array(all_list)[mpi_idx]

# --- start the process log --->
logout_fn = '_'.join(['proc',str(RANK).zfill(2)])+log_fn_out
if not logdir.exists(): logdir.mkdir(parents=True,exist_ok=True)
logout_fn = logdir/logout_fn

if save_log:
    fid = open(logout_fn,'w')
    txt_now   = datetime.now().strftime(format='%Y-%m-%dT%H:%M:%S')
    fid.write('=====================================================\n')
    fid.write('Rebuild database: {0}\n'.format(rebuild_db))
    fid.write('Database filenames: {0}\n'.format(coarse_fn))
    fid.write('Gridded granule data root: {0}\n'.format(gridroot))
    fid.write('---> Start regriding with processor: {0:02d} of {1:02d}\n'.format(RANK,NPROC))
    fid.write('Time now: {0}\n'.format(txt_now))
    fid.write('{0} month to process ...\n'.format(len(mon_idx)))
    fid.write('=====================================================\n')
    fid.close()



# --- start timer --- 
start_time = timeit.default_timer()
start_time0 = start_time

n_file = 0
for imon in mon_idx:
    t_start, t_end = time_interval(tt,imon)
    tdf_feat = coarse_df[(coarse_df.time>=t_start) & (coarse_df.time<=t_end)]#.iloc[range(2)]
    ngrns = len(tdf_feat)
    
    if ngrns>0:
        yrmontxt = t_start.strftime(format='%Y%m')
        outfn = '_'.join([PRD,yrmontxt])+'.nc'
        outfn = composite_dir/outfn
        tgrd = ggs(db_feature=tdf_feat)
        encoding = {var: fcomp for var in tgrd.data.data_vars}
        tgrd.data.to_netcdf(outfn,encoding=encoding)
        n_file += 1
        
        delta_time = timeit.default_timer() - start_time
        if save_log:
            fid = open(logout_fn,'a+')
            fid.write('\t{0} files for {1}...\n'.format(ngrns,yrmontxt)  )
            fid.write( '\tNumber of month processed:{0:7d}, time passed: {1:f}s\n'.format(n_file,delta_time) )
            fid.close()
        start_time = timeit.default_timer()
 
            
end_time = timeit.default_timer()  
if save_log:
    fid = open(logout_fn,'a+')    
    fid.write( '========================================================\n' )
    fid.write( 'Done. Total time used by processor {0}: {1}s for {2} files\n'.format(RANK,end_time-start_time0,n_file) )
    fid.write( '=========================================================\n' )
    fid.close()              
    
