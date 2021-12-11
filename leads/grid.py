import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import cartopy.crs as ccrs
import re

from typing import Union, Optional, List, Callable, Tuple #, Type

from leads.granule import Granule#, ATL09, ATL07, ATL10
from leads.database import is2db
from .utils import mapset,ptproj


Scalar = Union[int,np.int64,float]

class Grid:
    def __init__(self,grid_db=None,grid_root=None,prd='ATL07'):
        '''
        Setup the Grid instance. 
        Load the data from the monthly mean files if provided. 
        
        Parameters:
        ----------
        grid_db: optional, the pd.DataFrame containing the info of 
                 the monthly mean files. 
        grid_root: optional, the path to the monthly mean files.
        prd: optional, the product name of the data used to identify
             the sea ice features. 
        '''
        if grid_db is not None:
            self.db = grid_db
        elif grid_root is not None:
            self.db = self.grid_db(grid_root)
            
        if hasattr(self,'db'):
            self.load()
        return
    
    def load(self) -> None: 
        '''
        Load the monthly mean data from the files
        in the database db. 
        '''
        with xr.open_mfdataset(self.db.fn.values,combine='nested',concat_dim='time') as ds:
            ds['time'] = (['time'],self.db.time.values)
            self.data = ds
        return 
    
    @staticmethod
    def grid_db(grid_root: Union[str,Path],
                prd: str = 'ATL07',
               ) -> pd.DataFrame :
        '''
        Return the pd.DataFrame of the monthly mean files.
        
        Parameters:
        ----------
        grid_root: the root of the monthly mean files. 
        prd: optional, the name of the IS2 ATL data, str. 
        
        Return:
        ------
        the pd.DataFrame containing the info of the monthly mean files.
        '''
        pattern = 'ATL[0-9][0-9]_20[1-9][0-9][0-1][0-9].nc'.format(prd)
        fns = sorted( Path(grid_root).glob(pattern) )
        rx = re.compile('(ATL\\d{2})_(\\d{4})(\\d{2}).nc')
        dict_list = []
        for ifn,fn in enumerate(fns):
            PRD,YY,MM = rx.findall(fn.name).pop()
            tt = pd.to_datetime(YY+MM,format='%Y%m')
            tdict = dict(prd=PRD,fn=fn,year=int(YY),month=int(MM),time=tt)
            dict_list.append(tdict)
        tdf = pd.DataFrame(dict_list).drop_duplicates()
    
        return tdf
    
    @staticmethod
    def mean(ds: xr.Dataset,
             dim: Optional[Union[str,List]] = None,
            ) -> xr.Dataset:
        '''
        Return the mean for variables in the dataset.
        
        Parameters:
        ----------
        ds: xr.Dataset of input data
        dim: optional, name of dimension or a list of the names of 
             the dimensions along which to compute the mean
        
        Return:
        ------
        the xr.Dataset containing the mean variables
        '''
        if 'time' in dim: n_mon = len(ds.time)
        nds = Grid.apply_weighting(ds)
        nds = nds.sum(dim=dim)
        nds.attrs['n_mon'] = n_mon
        ods = Grid.normalize(nds)
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods
    
    @staticmethod
    def normalize(ds: xr.Dataset) -> xr.Dataset:
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
        feat_list = ['lead','floe','gap']
        for vn in ds.data_vars:
            if vn in ['lat','lon','n_granule']:
                ods[vn] = ds[vn]
            else:
                is_feat = False
                bool_wt = True
                for feat in feat_list:
                    if feat in vn:
                        is_feat = True
                        break
                if is_feat:                    
                    if 'n_' in vn:
                        weight = ds.attrs['n_mon']
                        bool_wt = False
                    elif 'length_' in vn:
                        weight = ds['n_'+feat]
                    else:
                        weight = ds['length_'+feat]
                else:
                    if vn=='nprof':
                        weight = 1.
                        bool_wt = False
                    else:
                        weight = ds['nprof'] 
                if bool_wt:
                    ods[vn] = ds[vn].where(
                        weight==0, ds[vn]/weight   ) 
                else:
                    ods[vn] = ds[vn]/weight
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods
    
    
    @staticmethod
    def apply_weighting(ds: xr.Dataset) -> xr.Dataset:
        '''
        Apply weighting of total length to data. 
        
        Parameters:
        ----------
        ds: xr.Dataste of the data to be weighted
        
        Return:
        ------
        xr.Dataset of the weighted data
        '''
        ods = xr.Dataset()
        feat_list = ['lead','floe','gap']
        for vn in ds.data_vars:
            if vn in ['lat','lon','n_granule']:
                ods[vn] = ds[vn]
            else:
                is_feat = False
                for feat in feat_list:
                    if feat in vn:
                        is_feat = True
                        break
                if is_feat:                    
                    if 'n_' in vn:
                        weight = 1.
                    elif 'length_' in vn:
                        weight = ds['n_'+feat]
                    else:
                        weight = ds['n_'+feat] * ds['length_'+feat]
                else:
                    if vn=='nprof':
                        weight = 1.
                    else:
                        weight = ds['nprof'] 
                ods[vn] = ds[vn] * weight
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods
    
    
    
class Grid_Coarse(Grid):
        
    
    def mean(self,
             ds: xr.Dataset,
             dim: Optional[Union[str,List]] = None,
            ) -> xr.Dataset:
        '''
        Return the mean for variables in the dataset.
        
        Parameters:
        ----------
        ds: xr.Dataset of input data
        dim: optional, name of dimension or a list of the names of 
             the dimensions along which to compute the mean
        
        Return:
        ------
        the xr.Dataset containing the mean variables
        '''
        if 'time' in dim: n_mon = len(ds.time)
        nds = self.apply_weighting(ds)
        nds = nds.sum(dim=dim)
        nds.attrs['n_mon'] = n_mon
        #return nds
        ods = self.normalize(nds)
        for vn in ds.attrs.keys():
            ods.attrs[vn] = ds.attrs[vn]
        return ods 
    
    def normalize(self,ds: xr.Dataset) -> xr.Dataset:
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
            if vn in ['lat','lon','n_granule'] or 'nprof_' in vn:
                ods[vn] = ds[vn]/ds.attrs['n_mon']
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
    
    
    def apply_weighting(self,ds: xr.Dataset) -> xr.Dataset:
        '''
        Apply weighting of total length to data. 
        
        Parameters:
        ----------
        ds: xr.Dataste of the data to be weighted
        
        Return:
        ------
        xr.Dataset of the weighted data
        '''
        ods = xr.Dataset()
        feat_list = ['lead','floe','gap','other','all']
        cpp_list = ['cth','cbh','cdp']
        for vn in ds.data_vars:
            if vn in ['lat','lon','n_granule'] or 'nprof_' in vn:
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
