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

from leads.granule import Granule
from leads.database import is2db
from leads.utils import mapset,ptproj

ZBOTMAX = 500
ZBOTMAX = [500,750,1000]
D_LEAD = [500,1000,2000]
prd_name = 'ATL07'
version = 'V004'
feat_res = '200x200'
beam_num = 3
rerun = False
rebuild_db = False
rebuild_feature = True
regrid_coarse = False
prdlist = ['ATL07','ATL09']
is2db_fn = 'IS2_{0}_{1}_{2}.db'.format(version,feat_res,prd_name)
# --- I/O directories ---
IS2root  = Path('/wrf/IS2')
dataroot = IS2root/version
featroot = IS2root/'feat'/version#/feat_res
gridroot = IS2root/'LAEA_granule'/version/feat_res
logdir= Path('./log_feat')
log_fn_ext = 'feat_regrid.log'

feat_dir = featroot/prd_name/str(beam_num)
grid_dir = gridroot/prd_name/str(beam_num)
if not feat_dir.exists(): feat_dir.mkdir(parents=True,exist_ok=True)
if not grid_dir.exists(): grid_dir.mkdir(parents=True,exist_ok=True)


corrupt_list = [
    (81,2) # ---> 
]
# --- nc compression setting ---
fcomp = dict(zlib=True, complevel=5, dtype='float32')

# --- mpi setting ---
PROC_ROOT = 0
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

# --- prepare the database ---
if rebuild_db:
    is2sub= is2db(dataroot=dataroot,prdlist=prdlist)
    is2sub.to_file(is2db_fn)
else:
    is2sub = is2db(filename=is2db_fn)

db_prd_org = getattr(is2sub,prd_name)
db_finished = pd.DataFrame()
db_prd = db_prd_org
if not rerun:
    
    if rebuild_feature:
        db_finished = is2db.build_db(
            featroot,'ATL07',beam=beam_num,subset='',
            filetype='csv',strong=False)
        db_prd = db_prd[~pd.to_datetime(db_prd.time).isin(db_finished.time.values)]        
    
        
    if regrid_coarse:
        db_coarse = is2db.build_db(
            gridroot,'ATL07',beam=beam_num,subset='',filetype='nc',strong=False)
        if len(db_coarse)>0:
            db_prd = db_prd[~pd.to_datetime(db_prd.time).isin(db_coarse.time.values)]
        
ngrns = len(db_prd)

mds,mapproj = mapset('LAEA',WIDTH=6600e3,dx=200e3,dy=200e3)

N0 = 0; N1 = len(db_prd)
all_list = range(N0,N1)
mpi_idx = list(range(RANK,len(all_list),NPROC))
file_idx = np.array(all_list)[mpi_idx]


irs = file_idx[0]
ire = file_idx[-1]

grn0 = db_prd.iloc[irs]
grn1 = db_prd.iloc[ire]

# --- start the process log --->
logfn = '_'.join(['proc',str(RANK).zfill(2),log_fn_ext])
logfn = logdir/logfn
if not logdir.exists(): logdir.mkdir(parents=True,exist_ok=True)

fid = open(logfn,'w')
txt_now   = datetime.now().strftime(format='%Y-%m-%dT%H:%M:%S')
fid.write('=====================================================\n')
fid.write('Rebuild database: {0}\n'.format(rebuild_db))
fid.write('Rebuild feature: {0}\n'.format(rebuild_feature))
fid.write('Regrid coarse method: {0}\n'.format(regrid_coarse))
fid.write('Database filename: {0}\n'.format(is2db_fn))
fid.write('IS2 data root: {0}\n'.format(IS2root))
fid.write('Feature data root: {0}\n'.format(featroot))
fid.write('Gridded cloud and feature data root: {0}\n'.format(gridroot))
fid.write('---> Start regriding with processor: {0:02d} of {1:02d}\n'.format(RANK,NPROC))
fid.write('Time now: {0}\n'.format(txt_now))
fid.write('{0} files to process ...\n'.format(len(file_idx)))
fid.write('From cycle {0:03d}, RGT {1:04d} at {2} ...\n'.format(grn0.cycle,grn0.rgt,grn0.time))
fid.write('To   cycle {0:03d}, RGT {1:04d} at {2} ...\n'.format(grn1.cycle,grn1.rgt,grn1.time))
fid.write('Plume base maximum height: {0} m.\n'.format(ZBOTMAX))
fid.write('=====================================================\n')
fid.close()


# --- start timer --- 
start_time = timeit.default_timer()
start_time0 = start_time

n_file = 0
irs = -1

for igran in file_idx:
    rgt = db_prd.iloc[igran].rgt
    cycle = db_prd.iloc[igran].cycle
    db_gran = is2sub.query(rgt=rgt,cycle=cycle)
    if (rgt,cycle) in corrupt_list: 
        print('RANK {0}'.format(RANK), 'Corrupted file: RGT {0}, cycle {1}'.format(rgt, cycle))
    if len(db_gran.ATL09)>0 and (rgt,cycle) not in corrupt_list:
        if irs<0: irs=igran
        # --- output filename ---
        fn07 = Path(getattr(db_gran,prd_name).iloc[0].fn)
        feat_fn = fn07.stem+'.csv'
        feat_fn = feat_dir/feat_fn
        # --- grid_coarse_fn is used with regrid_coase
        grid_coarse_fn = fn07.stem+'.nc'
        grid_coarse_fn = grid_dir/grid_coarse_fn
        # --- status check in ---
        if (igran-irs)%1==0: # and igran>irs:
            delta_time = timeit.default_timer() - start_time
            fid = open(logfn,'a+')
            fid.write( '\tNumber of files processed:{0:7d}, time passed: {1:f}s\n'.format(n_file,delta_time) )
            fid.write('\tCurrent file:{0}\n'.format(fn07.name)  )
            fid.close()
            start_time = timeit.default_timer()
            
        tgran = Granule(rgt=rgt,cycle=cycle, beam_num=beam_num,prdlist=prdlist,db=db_gran)
        
        # --- setup map and vertical coordinate
        if igran==irs:             
            izdict = tgran.ATL09.z_index()
        if rebuild_feature:
            tgran.save_feature(filename=feat_fn)
        else:
            tgran.load_feature(filename=feat_fn)
            
        if regrid_coarse:
            ods, regrid_stat, errmsg = tgran.regrid_coarse(
                ZBOT_PLUME=ZBOTMAX,D_LEAD=D_LEAD,
                lead_prd=prd_name,mds=mds,mapproj=mapproj,izdict=izdict)
            if regrid_stat==1:
                encoding = {var: fcomp for var in ods.data_vars}
                ods.to_netcdf(grid_coarse_fn,encoding=encoding)
                ods.close()    
            else:
                fid = open(logfn,'a+')
                fid.write('\t --->{0}\n'.format(errmsg))
                fid.write('\t Current file:{0}\n'.format(fn07.name)  )
                fid.write('\t RGT {0}, cycle {1}, beam {2}'.format(rgt,cycle,beam_num))
                fid.write('\t <--- skipped this granule\n')
                fid.close()            
                
 
            
        del tgran        
        
        n_file += 1
        if n_file%4==0: gc.collect()
            
            
end_time = timeit.default_timer()  
fid = open(logfn,'a+')    
fid.write( '========================================================\n' )
fid.write( 'Done. Total time used by processor {0}: {1}s for {2} files\n'.format(RANK,end_time-start_time0,n_file) )
fid.write( '=========================================================\n' )
fid.close()           

