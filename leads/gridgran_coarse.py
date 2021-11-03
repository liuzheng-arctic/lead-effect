import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import re

from typing import Union, Optional, List, NoReturn, Callable, Tuple #, Type

from leads.granule import Granule#, ATL09, ATL07, ATL10
from leads.gridgran import Gridded_Granule
from leads.database import is2db
from lzpyx.arccyc.cyctrack import mapset,ptproj


Scalar = Union[int,np.int64,float]


class GG_Coarse(Gridded_Granule):
    '''
    This class handles the IS2 cloud and feature statistics 
    in the LAEA grid produced using the gridded granule IS2 
    data. 
    
    parameters:
    ----------
    data: composite of IS2 gridded granule data on the full LAEA grid.
    mds:  xr.Dataset of map projection
    mapproj: handle to the cartopy.crs projection method.
    
    Methods:
    -------
    mapset          : adding the map projection mds and mapproj to instance.
    set_grid       : setup an empty xr.Dataset of the IS2 data on the LAEA grid.
    load_setup     : setup the empty data before loading the gridded granule data.
    clean_data     : clean up the data of this instance.
    load_granule   : add a list of gridded granule to the instance and normalize 
                     with the total feat length. 
    add_granule    : add another granule data to the existing data of instance.
                     Steps performed: apply weighting, accumulate
                     The resulting data is NOT normalized by the total feat length yet.
    apply_weighting: Apply the weighting of total feat length to the gridded 
                     granule data.
    ds_accumulate  : add weighted gridded granule data to the instance data.
                     The resulting dat is NOT normalized. 
    ds_normalize   : normalize the accumulated data with total feat length.
    feat_mean      : take the weighted average of the dataset in a feat length bin.
    grn2grid       : put gridded granule data into the full LAEA grid.     
    grn2grid_weighted* : put gridded granule data into the full LAEA grid and apply weighting.     
    
    * Depricated, to be removed.
    
    Examples:
    --------
    >>> feat_df = is2db.build_db(Path('/wrf/IS2/LAEA'),prd='ATL07',subset='',filetype='nc',fn_pre='feature',strong=False)
    >>> tdf = feat_df.iloc[range(2)]
    >>> tgrd = Gridded_Granule(db_feature=tdf)
    >>> mgrd = tgrd.feat_mean(varlist=['cld_all_lead'])
    '''
    
    def load_granule(self,
                     db_df: pd.DataFrame, 
                     db_landsea: Optional[pd.DataFrame] = None,
                     clean: Optional[bool] = True,
                    ) -> NoReturn:
        '''
        Load the gridded granule IS2 data to produce the composite.
        
        Parameters:
        ----------
        db_df: pd.DataFrame of the gridded granule IS2 database
        clean  : optional, boolean flag to clean the instance of data 
                 before loading, default True.
        '''        
        nfns = len(db_df)
        for igrn in range(nfns):
            fn = db_df.iloc[igrn].fn
            rgt = db_df.iloc[igrn].rgt
            cycle = db_df.iloc[igrn].cycle
            if igrn==0 and self.n_granule==-1:                
                self.load_setup(fn,clean=clean)
            print(igrn,fn)
            self.add_granule(fn)
        self.data = self.ds_normalize(self.data)
        return
        
    
        
    def set_grid(self,
                 ds: xr.Dataset,
                 ds_landsea: Optional[xr.Dataset] = None,
                ) -> xr.Dataset:
        '''
        Setup an empty xr.Dataset of the IS2 data on the LAEA grid.
        
        Parameters:
        ----------
        ds: xr.Dataset of a gridded granule IS2 data.
        
        Return:
        ------
        ods: an empty xr.Dataset on the LAEA grid defined by 
             the map projection of this instance. 
        '''
        if not hasattr(self,'mds') or not hasattr(self,'mapproj'):
            self.mapset(ds)
        ods = xr.Dataset()
        ods['lat'] = self.mds['lat']
        ods['lon'] = self.mds['lon']
        ods['Z'] = ds['Z']
        vns = list(ds.data_vars)
        
        for vn in vns:
            if vn not in ['iY','iX']:
                dims = [ x for x in ds[vn].dims if x!='GRID' ][::-1]
                vshp = [ds.dims[x] for x in dims] + list(ods.lat.shape) 
                ods[vn] = (dims+['Y','X'],np.zeros(vshp)) 
        ods['n_granule'] = (['Y','X'],np.zeros(ods.lat.shape))
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
            
        return ods
    
       
    def add_granule(self,
                    fn: Union[str,Path],
                    fn_landsea: Optional[Union[str,Path]] = None,
                   ) -> NoReturn:
        '''
        Add gridded granule IS2 data to the composite data on the full
        2D grid.
        
        Parameters:
        ----------
        fn: filename with full path to the gridded granule IS2 data file.         
        '''
        if not hasattr(self,'data'):
            self.load_setup(fn,clean=True)
        with xr.open_dataset(fn) as ds:
            nds = self.apply_weighting(ds)
            self.data = self.ds_accumulate(self.data,nds)   
            self.n_granule += 1         
        return nds
     
    def ds_normalize(self,ds: xr.Dataset, ds_type='feature') -> xr.Dataset:
        '''
        Normalize the weighted data with total feature length.
        
        Parameters:
        ----------
        ds: the weighted data in 2D grid to be accumulated upon.
        
        Return:
        ------
        The normalized data in 2D grid.
        '''
        ods = xr.Dataset()
        cpp_list = ['cth','cbh','cdp']
        for vn in ds.data_vars:
            if vn in ['iY','iX','lat','lon','n_granule'] or 'nprof_' in vn:
                ods[vn] = ds[vn]
            else:
                wrds = vn.split('_')
                vnprof = 'nprof_'+wrds[-1]
                weight = ds[vnprof]
                if wrds[0] in cpp_list:
                    vncld  = 'cld_'+wrds[1]+'_'+wrds[2]
                    weigth = ds[vnprof] * ds[vncld]                
                ods[vn] = ds[vn].where(
                    weight==0, ds[vn]/weight   ) 
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods
        

    def ds_accumulate(self,ds_0: xr.Dataset, ds_1: xr.Dataset) -> xr.Dataset:
        '''
        Accumulate a weighted new gridded granule data to the 
        un-normalized data in 2D grid. 
        
        Parameters:
        ----------
        ds_0: the un-normalized data in 2D grid to be accumulated upon.
        ds_1: the new weighted gridded granule to be accumulated. 
        
        Return:
        ------
        The updated un-normalized data in 2D grid.
        '''
        ny,nx = ds_0.lat.shape
        for vn in ds_1.data_vars:
            if vn not in ['lat','lon','iX','iY','n_granule']:
                for igrd in ds_1.GRID.values:
                    iy = int( ds_1.iY.values[igrd] )
                    ix = int( ds_1.iX.values[igrd] )
                    if ix<nx and iy<ny and ix>=0 and iy>=0:
                        ds_0[vn][...,iy,ix] += ds_1[vn].isel(GRID=igrd).values.T   
        for igrd in ds_1.GRID.values:
            iy = int( ds_1.iY.values[igrd] )
            ix = int( ds_1.iX.values[igrd] )
            if ix<nx and iy<ny and ix>=0 and iy>=0:
                ds_0['n_granule'][iy,ix] += 1
        return ds_0
    
    def apply_weighting(self,ds: xr.Dataset) -> xr.Dataset:
        '''
        Apply weighting of total length to data. 
        
        Parameters:
        ----------
        ds: xr.Dataste of the data (gridded granule or on full 2D grid) 
            to be weighted
        
        Return:
        ------
        xr.Dataset of the weighted data
        '''
        ods = xr.Dataset()
        feat_list = ['lead','floe','gap','other','all']
        cpp_list = ['cth','cbh','cdp']
        for vn in ds.data_vars:
            if vn in ['iY','iX','lat','lon','n_granule'] or 'nprof_' in vn:
                ods[vn] = ds[vn]
            else:
                wrds = vn.split('_')
                vnprof = 'nprof_'+wrds[-1]
                weight = ds[vnprof]
                if wrds[0] in cpp_list:
                    vncld  = 'cld_'+wrds[1]+'_'+wrds[2]
                    weigth = ds[vnprof] * ds[vncld]
                ods[vn] = ds[vn] * weight
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods