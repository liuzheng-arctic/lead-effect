import numpy as np
import pandas as pd
import xarray as xr
import h5py
from astropy.time import Time
import warnings
from pathlib import Path
from typing import Union, Optional, List


def strong_beam(t_stamp: pd.Timestamp) -> str:
    '''
    Find out the location of strong beams in the pair
    for a given time.
    Need to be updated after new yaw flip events.
    
    Parameters:
    ----------
    t_stamp: time of the granule using the filename
    
    Return:
    -------
    beam: the location, ['l','r'] of strong beam in pair, str
    '''
    t_yaw_flip =pd.to_datetime([
        '20181228-185251', '20190907-010350',
        '20200514-014903', '20210115-153410'],
        format='%Y%m%d-%H%M%S')
    yaw_flip = t_stamp>=t_yaw_flip
    beam = 'r'
    if any(yaw_flip):
        nflip = np.where(yaw_flip)[0][-1]
        if nflip%2==0: beam = 'l'            
    return beam

def read_beam_atl09(fn: Union[str,Path], beam: str, grp: Optional[str] = 'high_rate', 
                    varlist:Optional[ List[str] ] = None ,
                    **kwargs,
                   ) -> xr.Dataset:
    '''    
    Read data from a beam of IS2 ATL09 product. 
    The xr.open_dataset does not work on this dataset yet, likely because the second 
    dimension of surf_type is not defined anywhere in the dataset.     
    
    Parameters:
    ----------
    fn:  full path to the IS2 file, str or pathlib.Path object
    beam: beam id, str
    grp: the name of the group to be read, optional, default "high_rate". 
            Choose from []'high_rate','low_rate'].
    varlist: optional, list of str. The list of variable paths in the dataset. 
    
    Return: 
    -------
    xarray Dataset object of the IS2 data 
    '''
    
    if varlist is None: varlist = is2vars(fn)
    grp_path = '/'.join([beam,grp])
    
    attrlist = ['contentType', 'description', 'long_name', 'source', 'units',
                'flag_meanings', 'flag_values', 'valid_max', 'valid_min',
                '_FillValue', 'standard_name', 'CLASS', 'NAME', 'coordinates']
    dim_vns = ['delta_time','ds_layers','ds_va_bin_h','stype']
    ds = xr.Dataset()
    with h5py.File(fn,'r') as h5:
        # --- add dims to ds
        for vn in dim_vns:            
            vpath = '/'.join([grp_path,vn]) 
            if vpath in varlist: ds[vn] = h5[vpath][:]
        ds['stype'] = [1,2,3,4,5]     
        
        if len(ds.coords)>0:
            # --- add other variables
            #     (1) match dimensions
            #     (2) add attributes
            for vpath in varlist:
                if grp_path in vpath:
                    vn = vpath.split('/')[-1]
                    if vn in dim_vns:
                        pass
                    else:
                        ndim = h5[vpath].ndim
                        dims = []
                        vshape = h5[vpath].shape
                        for i_dim in range(ndim):
                            dimlen = vshape[i_dim]
                            for vdim in ds.coords:
                                if ds.coords[vdim].shape[0]==dimlen: dims.append(vdim)
                        ds[vn] = (dims,h5[vpath][:])
                    for vattr in h5[vpath].attrs.keys():
                        if vattr in attrlist: 
                            ds[vn].attrs[vattr] = h5[vpath].attrs[vattr]
    return ds

def read_beam_atl07(fn: Union[str,Path], beam: str, grplist: Optional[ List[str] ] = None,
                    si: Optional[bool] = True,
                    **kwargs,
                   ) -> xr.Dataset:
    '''
    Read data from a beam of IS2 ATL07 product. No geolocation group supported.
    This function does not check if the groups to be read exist in the file because
    it uses xr.open_dataset, which does not support group query.     
    
    Parameters:
    ----------
    fn:  full path to the IS2 file, str or pathlib.Path object
    beam: beam id, str
    grplist: a list of keywords for groups to be read, optional. Default includes
             ['heights','stats'].
    si: optional, boolean. /gtx/sea_ice_segments group will be read if True.
    
    Return: 
    -------
    xarray Dataset object of the IS2 data 
    '''
    grplist_default = ['heights','stats']
    drop_vars = [] 
    if 'drop_variables' in kwargs:
        drop_vars = kwargs['drop_variables']
    if grplist is None: grplist = grplist_default
    dslist = []
    for grp in grplist:
        grp_path = '/{0}/sea_ice_segments/{1}'.format(beam,grp)
        dslist.append( xr.open_dataset(fn,engine='h5netcdf',group=grp_path,decode_times=False,drop_variables=drop_vars) )
    # --- si has to read separately in order to add coords to other ds
    if si: 
        grp_path = '/{0}/sea_ice_segments'.format(beam)
        dssi = xr.open_dataset(fn,engine='h5netcdf',group=grp_path,decode_times=False)
        for i,ds in enumerate(dslist):
            dslist[i] = ds.assign_coords(dict(delta_time=dssi.delta_time))
        dslist.append(dssi)
    ods = xr.merge(dslist)
    for tds in dslist:
        tds.close()
    return ods

def read_beam_atl10(fn: Union[str,Path],beam: str,**kwargs) -> xr.Dataset:
    '''
    Read data from a beam of IS2 ATL07 product. 
    
    Only extract data under /{beam}/freeboard_beam_segment/height_segments
    and /{beam}/freeboard_beam_segment/beam_freeboard . 
    This assumes delta_time is included in the beam_freeboard group when 
    downloading the data. 
    The current code is tested with no coords in the height_segments group. 
    This function does not check if the groups to be read exist in the file because
    it uses xr.open_dataset, which does not support group query.     
    
    Parameters:
    ----------
    fn:  full path to the IS2 file, str or pathlib.Path object
    beam: beam id, str
    
    Return: 
    -------
    xarray Dataset object of the IS2 data 
    '''
    htpath = '/{0}/freeboard_beam_segment/height_segments'.format(beam)
    fbpath = '/{0}/freeboard_beam_segment/beam_freeboard'.format(beam)
    
    # --- read data in case no coords were written to this group
    fbds = xr.open_dataset(fn,engine='h5netcdf',group=fbpath,
                           decode_times=False,
                         backend_kwargs={'phony_dims':'access'})
    htds = xr.open_dataset(fn,engine='h5netcdf',group=htpath,
                           decode_times=False,
                         backend_kwargs={'phony_dims':'access'})
    # --- replace phony_dim with delta_time of the other group
    if 'delta_time' in list(htds.dims) + list(fbds.dims):
        if 'delta_time' not in htds.dims:
            dimvn = list(htds.dims)[0]
            htds = htds.rename({dimvn:'delta_time'})
            htds['delta_time'] = fbds.delta_time.values
        if 'delta_time' not in fbds.dims:
            dimvn = list(fbds.dims)[0]
            htds = htds.rename({dimvn:'delta_time'})
            htds['delta_time'] = fbds.delta_time.values
    else:
        raise ValueError('Need delta_time in height_segments or beam_freeboard.')
    ods = xr.merge([fbds,htds])
    fbds.close()
    htds.close()
    return ods


def read_beam(fn: Union[str,Path],beam: str,prd:str,**kwargs) -> xr.Dataset:
    '''
    Read data from a beam of IS2 product
    
    Parameters:
    ----------
    fn:  full path to the IS2 file, str or pathlib.Path object
    beam: beam id, str
    prd : product name, str, from ['ATL09','ATL10']
    
    Return:
    -------
    xarray Dataset object of the IS2 data 
    '''
    assert (prd in ['ATL07','ATL09','ATL10']), '{0} not supported'.format(prd)
    if prd=='ATL07': 
        return read_beam_atl07(fn,beam,**kwargs)
    elif prd=='ATL09':
        return read_beam_atl09(fn,beam,**kwargs)
    elif prd=='ATL10':
        return read_beam_atl10(fn,beam,**kwargs)



def is2time(delta_time,atlas_epoch=1198800018.0):
    '''
    Convert IS2 delta time to pandas datetime 
    
    Parameters:
    ----------
    delta_time: IS2 delta time, seconds since atlas_epoch
    atlas_epoch: Number of GPS seconds between the GPS epoch (1980-
                 01-06T00:00:00 Z UTC) and the ATLAS Standard Data
                 Product (SDP) epoch (2018-01-01:T00.00.00 UTC).
    
    Return:
    ------
    The ISO time in a pandas DatetimeIndex
    '''
    leapSecondsOffset=37
    gps_seconds = atlas_epoch + delta_time - leapSecondsOffset
    tgps = Time(gps_seconds, format='gps')
    tiso = Time(tgps, format='datetime')
    return pd.to_datetime(tiso.value)

def is2vars(fn):
    '''
    Get a list of data paths in an IS2 data file
    '''
    vlist = []
    vgrp = dict()
    
    def h5walk(name, node):
        if isinstance(node, h5py.Dataset):
            vlist.append(name)
        return 
    
    with h5py.File(fn,'r') as h5pt:
        h5pt.visititems(h5walk)
    return vlist

def is2vgrp(fn):
    '''
    Build a dictionary of the variable group from an IS2 data file
    
    Parameters:
    ----------
    fn: full path to the IS2 h5 file
    
    Return:
    ------
    vgrp: a dictionary of the data variables in IS2 data file
          The keys are the core data variable name and the values
          are the full path to this variable in different groups.
    '''
    vlist = []
    vgrp = dict()
    
    def h5walk(name, node):
        if isinstance(node, h5py.Dataset):
            vlist.append(name)
        return 
    
    with h5py.File(fn,'r') as h5pt:
        h5pt.visititems(h5walk)
        for vn in vlist:
            vkey = Path(vn).stem
            if vkey not in vgrp.keys():
                vgrp[vkey] = [vn]
            else:
                vgrp[vkey] += [vn]
    return vgrp

def is2path(paths,vtags):
    '''
    Find paths that contains all of vtags
    
    Parameters:
    ----------
    paths: list of str, hdf5 paths
    vtags: list of str, a list of subdirectory names of hdf5 paths.
    
    Return:
    ------
    opaths: list of str, a list of matching subset of paths    
    '''
    opaths = []
    for vpath in paths:
        pwords = vpath.split('/')
        if all( np.isin(vtags,pwords) ):
            opaths.append(vpath)
            
    if 'ds_surf_type' in paths:
        opaths = paths
    
    return opaths 

def beam_avail(prd):
    '''
    Return a list of available beam ids for the input prd.
    '''
    if prd=='ATL09':
        beam_list = ['profile_'+str(i+1) for i in range(3)]
    else:
        beam_list = ['gt'+str(i+1)+'r' for i in range(3)] 
        beam_list = beam_list + ['gt'+str(i+1)+'l' for i in range(3)] 
    return beam_list

def fn2rgt(filename):
    '''
    Extract rgt and cycle_number from IS2 ATL0x data standard filename.
    
    Parameters:
    ----------
    filename: full path with standard filename (including NSIDC processed) 
    
    return:
    ------
    rgt: reference ground track number (int)
    cyc: cycle number
    '''
    sfn = Path(filename).stem
    wrds = sfn.split('_')
    irgt = 3 if wrds[0]=='processed' else 2
    rgttxt = wrds[irgt]
    rgt = int(rgttxt[:4])
    cyc = int(rgttxt[4:6])
    return rgt,cyc

def ptproj(prj_in,prj_out,x_in,y_in):
    '''
    Convert the coordinates of input projection to the output projection
    
    Parameters:
    ----------
    prj_in : the map projection of the input coordinates
    prj_out: the map projection of the output coordinates
    x_in   : x coordinates in prj_in
    y_in   : y coordinates in prj_in
    #list   : A list of points or a single point, boolean, default True.
    
    Returns:
    -------
    x_out: x coordinates in prj_out
    y_out: y coordinates in prj_out
    '''
    if isinstance(x_in,(list,pd.core.series.Series,np.ndarray)):
        pts = prj_out.transform_points(prj_in,x_in,y_in)
        x_out = pts[...,0]; y_out = pts[...,1]
    else:
        x_out, y_out = prj_out.transform_point(x_in, y_in, prj_in)
        
    return x_out,y_out

#--------------------------------------------------
def mapset( pjname,lon0=315,lat0=90,WIDTH=6700e3,HEIGHT=7000e3,
           dx = 100e3, dy = 100e3,proj_only=False,*kwargs ):
#--------------------------------------------------
    '''
    Set the LAEA map for AC tracking and return grid info.

    Parameters:
    ----------
    pjname: short Name for the projection 
            e.g., LAEA for LambertAzimuthalEqualArea
    lon0:   central longitude
    lat0:   central latitude
    WIDTH:  width of domain [m]
    HEIGHT: height of domain [m]
    dx:     grid box width [m]
    dy:     grid box height [m]
    dy:     grid box height [m]
    proj_only: generate the projection only, 
            without saving the mapping of the ArcCyc grid.
    kwargs: dictionary of additional arguments

    Return:
    mds: an xarray Dataset with map info
    mapprj: Cartopy map projection object
    '''
    pcproj = ccrs.PlateCarree()
    if pjname=='LAEA':
        pj_longname = 'LambertAzimuthalEqualArea'
        prjhdl = getattr( ccrs , pj_longname )
        mapprj = prjhdl(
            central_longitude=lon0,central_latitude=lat0
        )
    if proj_only: return mapprj

    XLIM = WIDTH/2.
    YLIM = HEIGHT/2.
    nx   = int(WIDTH/dx)
    ny   = int(HEIGHT/dy)
    xgrid = np.linspace(-XLIM,XLIM,nx+1)
    ygrid = np.linspace(-YLIM,YLIM,ny+1)
    
    gridc = lambda x: (x[1:]+x[:-1])/2.
    xc = gridc(xgrid); yc = gridc(ygrid)
    xcs,ycs = np.meshgrid(xc,yc)
    pts = pcproj.transform_points(mapprj,xcs,ycs)
    lonc = pts[...,0]; latc = pts[...,1]
    
    
    mds = xr.Dataset()
    mds['X'] = xc
    mds['Y'] = yc
    mds['lat'] = (['Y','X'],latc)
    mds['lon'] = (['Y','X'],lonc)
    
    mds.attrs['WIDTH' ] = WIDTH
    mds.attrs['HEIGHT'] = HEIGHT
    mds.attrs['XLIM'  ] = XLIM
    mds.attrs['YLIM'  ] = YLIM
    mds.attrs['dx'    ] = dx
    mds.attrs['dy'    ] = dy
    mds.attrs['proj'  ] = pjname
    mds.attrs['proj_f'] = pj_longname
    mds.attrs['lat0'  ] = lat0
    mds.attrs['lon0'  ] = lon0

    return mds , mapprj

