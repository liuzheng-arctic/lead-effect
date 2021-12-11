import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

from typing import Union, Optional, List, Callable, Tuple,Type #, Type

from .utils import strong_beam, read_beam
from .leads import Lead, Floe
<<<<<<< HEAD
from .database import is2db
from lzpyx.arccyc.cyctrack import mapset, ptproj
=======
from .utils import mapset, ptproj
>>>>>>> 98382a0a4c62585bb2dabd2c93ec785b96db0afa

InDex = Union[int,np.int64,float]
IntType = Union[int, np.int64]
InDexList = Union[ List[InDex], np.ndarray ]

#class Granule(AttrDict):
class Granule():
    def __init__(self,
                 cycle: Optional[ IntType ] = None,
                 rgt: Optional[IntType ]= None,
                 beam_num: Optional[IntType ] = None,
                 prdlist: Optional[ List[str] ] = None,
                 db: Optional[ Type[is2db] ] = None,
                ) -> None:
        '''
        Initialize the ATL10 instance.
        
        Parameters:
        ----------
        rgt: optional, the RGT number of the granule, int
        cycle: optional, the cycle number of the granule, int
        beam: optional, the beam name, str
        data: optional, the ATL10 data, xr.Dataset
        db: optional, the database record of this granule, is2db class object
        '''
        if cycle is not None: self.cycle = cycle
        if rgt is not None: self.rgt = rgt
        if beam_num is not None: self.beam_num = beam_num
        if prdlist is not None: self.prdlist = prdlist
        if rgt is not None and cycle is not None and db is not None:
            self.db = db.query(rgt=rgt,cycle=cycle)
            if len(prdlist)!=0:
                for prd in prdlist: 
                    if prd in self.db.prdlist:
                        self.add_product(prd)
                    
        return
    

    def set_granule(self,
                    cycle: Optional[IntType ] = None,
                    rgt: Optional[IntType ]= None,
                    beam_num: Optional[IntType ] = None,
                    prdlist: Optional[ List[str] ] = None,
                    db: Optional[ Type[is2db] ] = None) -> None:
        '''
        Set granule attributes.
        
        Parameters:
        ----------
        rgt: optional, the RGT number of the granule, int
        cycle: optional, the cycle number of the granule, int
        beam_num: optional, the beam number, str
        prdlist: optional, a list of product names
        db: optional, the database record of this granule, is2db class object
        
        Parameter:
        '''
        if cycle is not None and not hasattr(self,'cycle'): self.cycle = cycle
        if rgt is not None and not hasattr(self,'rgt'): self.rgt = rgt
        if beam_num is not None and not hasattr(self,'beam_num'): self.beam_num = beam_num
        if prdlist is not None and not hasattr(self,'prdlist'): self.prdlist = prdlist
        if db is not None and not hasattr(self,'db'): self.db = db
        return
    
    def info(self) -> None:
        '''
        Print out information of the granule. 
        '''
        nospec = 'Not specified'
        rgt = self.rgt if hasattr(self,'rgt') else nospec
        cycle = self.cycle if hasattr(self,'cycle') else nospec
        beam = self.beam_num if hasattr(self,'beam_num') else nospec
        prdlist = self.prdlist if hasattr(self,'prdlist') else nospec
        infomsg = 'RGT: {0}, cycle: {1}, beam number: {2}, products: {3}'.format(
            rgt,cycle,beam,prdlist)
        print(infomsg)
        print('------>')
        print('The information of the database of this granule:')
        if hasattr(self,'db'): self.db.disp_gran(rgt=rgt,cycle=cycle)
        return
    
    def add_product(self,prd: str) -> None:
        '''
        Add a data product to the granule.
        
        Parameters:
        ----------
        prd: the product name, str
        '''
        errmsg_all = 'Need rgt, cycle, beam_num, and database for Granule to add product.'
        assert all( v is not None for v in [self.rgt, self.cycle, self.beam_num, self.db]), errmsg_all
        beam_num = self.beam_num
        rgt = self.rgt
        cycle = self.cycle
        errmsg_none = 'No granule found for rgt {0}, cycle {1}'.format(rgt,cycle)
        errmsg_mult = 'Multiple granule found for rgt {0}, cycle {1}'.format(rgt,cycle)
        
        t_df = getattr(self.db,prd)
        if len(t_df)==0: 
            raise ValueError(errmsg_none)
        elif len(t_df)>1:
            raise ValueError(errmsg_mult)
        else:
            t_prd = t_df.iloc[0].time
            if prd == 'ATL09':
                beam = 'profile_'+str(beam_num)
            else:
                beam = 'gt'+str(beam_num)+strong_beam(t_prd)
            fn = t_df.iloc[0].fn
            kwargs = {'drop_variables':['hist_w','height_segment_rms']} if prd=='ATL07' else {}
            dat = read_beam(fn,beam,prd,**kwargs)
            
            if prd =='ATL10':
                self.ATL10 = ATL10(rgt,cycle,beam=beam,data=dat)
            elif prd=='ATL09':
                self.ATL09 = ATL09(rgt,cycle,beam=beam,data=dat)
            elif prd=='ATL07':
                self.ATL07 = ATL07(rgt,cycle,beam=beam,data=dat)
            else:
                setattr(self,prd,dat)
                
            setattr(getattr(self,prd),'beam',beam)
        
        return
    

    def regrid_coarse(self, 
                    mds: Optional[xr.Dataset] = None,
                    mapproj: Optional[callable] = None, 
                    grid_name: Optional[str] = 'LAEA',
                    lead_prd: Optional[str] = 'ATL07',
                    dT: Optional[float] = .02,
                    D_LEAD: Optional[Union[int,float]] = 1000,
                    izdict: Optional[dict] = None,
                    ZBOT: Optional[Union[int,float]] = -100,
                    ZTOP: Optional[Union[int,float]]= 14e3,
                    ZLOW: Optional[Union[int,float]] = 2e3,
                    ZMID: Optional[Union[int,float]] = 6e3,
                    ZBOT_PLUME: Optional[Union[int,float]] = 500,
              ) -> xr.Dataset :
        '''
        Regrid IS2 granule data on to a Polar LAEA grid. 
        Cloud and plume statistics over features by feature size in each gridbox are 
        computed and stored in xr.Dataset. 
        The features include leads, floes, and the gaps in lead data (ATL07/ATL10). 
        Cloud statistics and features are weighted by fractional length in each
        grid box corresponding to the features. 
        Plume is defined as continuous cloud of the lowest level with cloud base 
        below ZBOTMAX. 
        
        Parameters:
        ----------
        mds: optional, the xr.Dataset containing the map projection info defined by 
             mapset from lzpyx.arccyc.cyctrack. Default None
        mapproj: optional, the cartopy.crs map projection method used to project data,
                 default None.
        grid_name: optional,str, projection grid name, default "LAEA".
        lead_prd : optiona, str, lead product name, default "ATL07".
        dT  : optional, float, the extra time [s] allowed before and after the feature timestamp 
              used to find collocated ATL09 cloud profiles. Default 0.02, the half width of
              the ATL09 high rate profile time increment. 
        D_LEAD: optional, float, lead plume search distance in meters. 
        izdict: optional, dictionary, containing the vertical indices for cloud in different 
                height levels with keywords in ['all','low','mid','high'].
        ZBOT: optional, float, the bottom altitude [m] used to subset in the vertical, default -100.
        ZTOP: optional, float, the top altitude [m] used to subset in the vertical, 
              default 14 km. IS2 altitude is folded every 15 km, starting from -900+ m. 
        ZLOW: optional, float, the altitude [m] used to define the top of 
              low level clouds, default 2 km. 
        ZMID: optional, float, the altitude [m] used to define the top of 
              middle level clouds, default 6 km. 
        ZBOT_PLUME: optional, float, the altitude [m] used as the highest low cloud plume, 
                 default, 500m.    
        
        Return:
        ------
        ods: xr.Dataset containing the cloud, plume, and feature statistics. 
             FEAT in ['lead','floe','gap']; LVL in ['all','low','mid','high']
             The variables in ods: 
             n_FEAT      :: [FEAT_LBND,GRID], the fractional total of FEAT in grid box and width bin. 
             length_FEAT :: [FEAT_LBND,GRID], the weighted mean of length of FEAT in grid box and width bin. 
             cldfrc_FEAT :: [FEAT_LBND, GRID, NZ], the mean cloud fraction profile over FEAT 
                            in grid box and width bin.
             cldfrc_plume_FEAT :: [FEAT_LBND, GRID, NZ], the mean cloud fraction profile of 
                                  plumes over FEAT in grid box and width bin.
             cld_LVL_FEAT   :: [FEAT_LBND,GRID], the shaded cloud fraction at level LVL, 
                               in grid box and width bin.      
             cld_plume_FEAT :: [FEAT_LBND,GRID], the shaded plume cloud fraction at level LVL, 
                               in grid box and width bin.                                           
        '''
        if isinstance(ZBOT_PLUME,(int,float,np.int64)):
            zbot_plm = np.array([ZBOT_PLUME])
        elif isinstance(ZBOT_PLUME,(list,np.ndarray)):
            zbot_plm = ZBOT_PLUME
            
        if isinstance(D_LEAD,(int,float,np.int64)):
            lead_dist = np.array([0,D_LEAD])
        elif isinstance(D_LEAD,(list,np.ndarray)):
            lead_dist = np.append(0,np.array(D_LEAD))
        lead_dT = lead_dist/284*.04
    
        # --- set map info ---
        if mds is None and mapproj is None:
            mds, mapproj = mapset(grid_name)
        XLIM = mds.XLIM
        YLIM = mds.YLIM 
        pcproj = ccrs.PlateCarree()
        # --- check if lead data is loaded
        errmsg = 'No data for {0} is loaded to granule.'.format(lead_prd)
        if not hasattr(self,lead_prd): raise ValueError(lead_prd)
            
        lead_var = getattr(self,lead_prd)        
        # --- Find the sea ice data within domain
        dsfeat = lead_var._data
        lon = dsfeat.longitude.values
        lat = dsfeat.latitude.values
        xx,yy = ptproj(pcproj,mapproj,lon,lat)
        dsfeat['X'] = (['delta_time'],xx)
        dsfeat['Y'] = (['delta_time'],yy)
        tt07 = dsfeat.delta_time.where( 
            (dsfeat.X>=-XLIM) & (dsfeat.X<=XLIM) &
            (dsfeat.Y>=-YLIM) & (dsfeat.Y<=YLIM) , 
            drop=True)
        t00 = tt07.values[0]
        t11 = tt07.values[-1]
        
        # --- Find the cloud data in domain
        ds09_org = self.ATL09._data
        lon = ds09_org.longitude.values
        lat = ds09_org.latitude.values
        xx,yy = ptproj(pcproj,mapproj,lon,lat)
        ds09_org['X'] = (['delta_time'],xx)
        ds09_org['Y'] = (['delta_time'],yy)
        ds09 = ds09_org.where( 
            (ds09_org.X>=-XLIM) & (ds09_org.X<=XLIM) &
            (ds09_org.Y>=-YLIM) & (ds09_org.Y<=YLIM) , 
            drop = True )
        # --- find the index location of grid boxes
        ixs = np.floor((ds09.X.values+XLIM)/mds.dx)
        iys = np.floor((ds09.Y.values+YLIM)/mds.dy)
        ds09['iY'] = iys
        ds09['iX'] = ixs        
        iyx_list = sorted(set(zip(iys,ixs)))
        iyx_arr = np.array(iyx_list)
        
        zz  = ds09.ds_va_bin_h.values
        if izdict is None: 
            izdict = ATL09._z_index(zz,ZBOT=ZBOT,ZTOP=ZTOP,ZMID=ZMID,ZLOW=ZLOW)
        
        nplm = len(zbot_plm)
        ndlead = len(lead_dist)
        nz  = len(izdict['all'])
        ngrid = len(iyx_list)
         
        ods = xr.Dataset()
        ods['Z'] = zz[izdict['all']]
        ods['GRID'] = range(ngrid)
        ods['ZBOT_PLUME'] = zbot_plm
        ods['D_LEAD'] = lead_dist
        ods['iY'] = (['GRID'],iyx_arr.T[0])
        ods['iX'] = (['GRID'],iyx_arr.T[1])
        
        lvl_list = ['all','low','mid','high']
        vnpre_list = [ 'cld_'+lvl+'_' for lvl in lvl_list ] 
        cpp_list = ['cth','cbh','cdp']
        dim3d = ['GRID','Z']
        shp3d = (ngrid,nz)  
        dim_plm = ['GRID','Z','ZBOT_PLUME']
        shp_plm = (ngrid,nz,nplm) 
        feat_list = ['floe','gap']
        nonf_list = ['other','all']
    
        for feat in feat_list+nonf_list:
            ods['cldfrc_'+feat] = ( dim3d, np.zeros(shp3d) )
            ods['cldfrc_plume_'+feat] = ( dim_plm, np.zeros(shp_plm) )
            for vnpre in vnpre_list + ['nprof_']:
                ods[vnpre+feat] = ( ['GRID'], np.zeros(ngrid) )
            for cpp in cpp_list:
                ods['cld_plume_'+feat] = ( ['GRID','ZBOT_PLUME'], np.zeros((ngrid,nplm)) )
                ods[cpp+'_all_'+feat] = ( ['GRID'], np.zeros(ngrid) )
                ods[cpp+'_plume_'+feat] = ( ['GRID','ZBOT_PLUME'], np.zeros((ngrid,nplm)) )    
            ods['cldfrc_'+feat].attrs['long_name'] = 'Mean cloud fraction profile over {0}.'.format(feat) 
            ods['cldfrc_plume_'+feat].attrs['long_name'] = 'Mean cloud plume fraction profile over {0}.'.format(feat) 
            
        feat = 'lead'
        ods['cldfrc_'+feat] = ( ['GRID','D_LEAD','Z'], np.zeros((ngrid,ndlead,nz)) )
        ods['cldfrc_plume_'+feat] = ( ['GRID','D_LEAD','Z','ZBOT_PLUME'], np.zeros((ngrid,ndlead,nz,nplm)) )
        for vnpre in vnpre_list +['nprof_']:
            ods[vnpre+feat] = ( ['GRID','D_LEAD'], np.zeros((ngrid,ndlead)) )
        for cpp in cpp_list:
            ods['cld_plume_'+feat] = ( ['GRID','D_LEAD','ZBOT_PLUME'], np.zeros((ngrid,ndlead,nplm)) )
            ods[cpp+'_all_'+feat] = ( ['GRID','D_LEAD'], np.zeros((ngrid,ndlead)) )
            ods[cpp+'_plume_'+feat] = ( ['GRID','D_LEAD','ZBOT_PLUME'], np.zeros((ngrid,ndlead,nplm)) )   
            
        ods.attrs['ZBOT'] = ZBOT
        ods.attrs['ZLOW'] = ZLOW
        ods.attrs['ZMID'] = ZMID
        ods.attrs['ZTOP'] = ZTOP
        for vn in mds.attrs.keys():
            ods.attrs[vn] = mds.attrs[vn]
       
        # --- combine features into one DataFrame
        if not hasattr(self,'all_feature'):
            self.load_feature(prd_name=prdname,filename=filename)
        dfall = self.all_feature 
        
        success = 0
        for igrd in range(ngrid):             
            iy,ix = iyx_arr[igrd].astype(int)
            fgrd = (ds09.iY.values==iy) & (ds09.iX.values==ix)
            if fgrd.any():
                idx = np.where(fgrd)[0]
                i0 = idx[0]
                i1 = idx[-1]
                tt_grd = ds09.delta_time.values[fgrd]
                t0_grd = tt_grd[0]
                t1_grd = tt_grd[-1]
                nprof = len(tt_grd)
                if nprof!=i1-i0+1:
                    success = -1
                    errmsg = 'Kinks in geolocation. '
                    return ods, success, errmsg
                # --- 
                cmdict = ATL09.cloudmask(ds09,i0=i0,i1=i1,plume=True,shade=True,
                                         izdict=izdict,ZBOT_PLUME=ZBOT_PLUME)
                # --- Feat flag: 
                #     0: other; 1: lead; 2: floe; 3: gap
                feat_flag = np.zeros(nprof)
                lead_flag = np.zeros((nprof,ndlead))
                feat_grd = dfall[(dfall.t_start>=t0_grd-dT) & 
                                 (dfall.t_end<=t1_grd+dT)]
                # --- get the feat flag for each profile
                if len(feat_grd)>0: 
                    for i_prof in range(nprof):
                        tdf = feat_grd[(feat_grd.t_start<tt_grd[i_prof]) &
                                       (feat_grd.t_end>tt_grd[i_prof])]
                        if len(tdf)>0:
                            feat_i_prof = tdf.iloc[0].feat
                            if feat_i_prof=='lead':
                                feat_flag[i_prof] = 1
                            elif feat_i_prof=='floe':
                                feat_flag[i_prof] = 2
                            elif feat_i_prof=='gap':
                                feat_flag[i_prof] = 3
                    for i_feat in range(len(feat_grd)):
                        if feat_grd.iloc[i_feat].feat=='lead':
                            t0_feat_i = feat_grd.iloc[i_feat].t_start
                            t1_feat_i = feat_grd.iloc[i_feat].t_end
                            for i_dlead in range(ndlead):
                                dT_i = lead_dT[i_dlead]
                                f_feat = np.logical_and(
                                    tt_grd>t0_feat_i-dT_i,
                                    tt_grd<t1_feat_i+dT_i )
                                if f_feat.any():
                                    lead_flag[f_feat,i_dlead] = 1
                                    fnd_lead = True
                                
                
                for feat in feat_list+nonf_list:
                    if feat=='lead':
                        feat_i = 1
                    elif feat=='floe':
                        feat_i = 2
                    elif feat=='gap':
                        feat_i = 3
                    elif feat=='other':
                        feat_i = 0
                    if feat=='all':
                        fdx_feat = np.array([True for x in feat_flag])
                    else:
                        fdx_feat = feat_flag==feat_i
                    if fdx_feat.any() and feat!='lead':
                        nprof_feat = len(np.where(fdx_feat)[0])
                        ods['nprof_'+feat][igrd] = nprof_feat
                        ods['cldfrc_'+feat][igrd] = cmdict['CM'][fdx_feat].sum(axis=0)[izdict['all']]/nprof_feat
                        ods['cldfrc_plume_'+feat][igrd] = cmdict['CM_plume'][fdx_feat].sum(axis=0)[izdict['all']]/nprof_feat
                        for lvl in lvl_list+['plume']:
                            ods['cld_'+lvl+'_'+feat][igrd] = cmdict['cld_'+lvl][fdx_feat].sum(axis=0)/nprof_feat
                        for cpp in cpp_list:
                            ods[cpp+'_all_'+feat][igrd] = cmdict[cpp][fdx_feat].sum(axis=0)/nprof_feat 
                            ods[cpp+'_plume_'+feat][igrd] = cmdict[cpp+'_plume'][fdx_feat].sum(axis=0)/nprof_feat  
                feat = 'lead'            
                for i_dlead in range(ndlead):
                    fdx_feat = lead_flag[:,i_dlead]==1
                    if fdx_feat.any():
                        nprof_feat = len(np.where(fdx_feat)[0])
                        ods['nprof_'+feat][igrd,i_dlead] = nprof_feat
                        ods['cldfrc_'+feat][igrd,i_dlead] = cmdict['CM'][fdx_feat].sum(axis=0)[izdict['all']]/nprof_feat  
                        ods['cldfrc_plume_'+feat][igrd,i_dlead] = cmdict['CM_plume'][fdx_feat].sum(axis=0)[izdict['all']]/nprof_feat
                        for lvl in lvl_list+['plume']:
                            ods['cld_'+lvl+'_'+feat][igrd,i_dlead] = cmdict['cld_'+lvl][fdx_feat].sum(axis=0)/nprof_feat
                        for cpp in cpp_list:
                            ods[cpp+'_all_'+feat][igrd,i_dlead] = cmdict[cpp][fdx_feat].sum(axis=0)/nprof_feat 
                            ods[cpp+'_plume_'+feat][igrd,i_dlead] = cmdict[cpp+'_plume'][fdx_feat].sum(axis=0)/nprof_feat  
        success = 1
        errmsg = 'Finished.'
        return ods, success, errmsg
    
    
    def load_feature(self,lead_prd='ATL07',filename=None): 
        '''
        Load granule features from file.
        '''
        # --- combine features into one DataFrame
        if filename is None:
            lead_var = getattr(self,lead_prd)
            df_lead = lead_var.leads
            df_floe = lead_var.floes
            df_gap  = lead_var.locate_gap(lead_var._data)        
            df_lead['feat'] = 'lead'
            df_floe['feat'] = 'floe'
            df_gap['feat']  = 'gap'   
            dfall = pd.DataFrame()
            if len(df_lead)>0:
                dfall = dfall.append(df_lead,ignore_index=True)
            if len(df_floe)>0:
                dfall = dfall.append(df_floe,ignore_index=True)
            if len(df_gap)>0:
                dfall = dfall.append(df_gap,ignore_index=True)
            dfall = dfall.sort_values(by='sid',ignore_index=True)
            self.all_feature = dfall
            getattr(self,lead_prd)._gaps = df_gap
        else:
            dfall = pd.read_csv(filename)
            tleads = dfall[dfall['feat']=='lead']
            tfloes = dfall[dfall['feat']=='floe']
            tgaps = dfall[dfall['feat']=='gap']
            self.all_feature = dfall
            if len(tleads)>0: getattr(self,lead_prd)._leads = tleads
            if len(tfloes)>0: getattr(self,lead_prd)._floes = tfloes
            if len(tgaps)>0:  getattr(self,lead_prd)._gaps = tgaps
        return 
    
    def save_feature(self,filename,lead_prd='ATL07',compression=None):
        '''
        Save features in granule to file.
        '''
        if not hasattr(self,'all_feature'):
            self.load_feature(lead_prd=lead_prd)
        self.all_feature.to_csv(filename,index=False,compression=compression)
        
        return
    
    @staticmethod
    def collocate(ds: xr.Dataset,
                  t_start: Optional[Union[float,pd.Timestamp]] = None,
                  t_end: Optional[Union[float,pd.Timestamp]] = None
                 ) -> Tuple[bool,InDex,InDex]:
        '''
        Find the start and end indice of dataset 
        collocating to the start and end time.
        
        Parameters:
        ----------
        ds: xr.Dataset of the IS2 dataset
        t_start: optional, starting delta_time, float or pd.Timestamp
        t_end: optional, ending delta_time, float or pd.Timestamp
        '''
        tt = ds.delta_time.values
        if t_start is None: t_start = tt[0]
        if t_end is None: t_end = tt[-1]
        fdx = np.logical_and(tt>=t_start, tt<=t_end)
        fnd = any(fdx)
        i0 = -1
        i1 = -1
        if fnd:
            idx = np.where(fdx)[0]
            i0 = idx[0]
            i1 = idx[-1]
        return fnd, i0,i1
        
    
    
class ATL09():
    def __init__(self,
                 rgt:IntType = None,
                 cycle:IntType = None,
                 beam: Optional[str] = None,
                 data: Optional[xr.Dataset] = None) -> None:
        '''
        Initialie the ATL10 instance.
        
        Parameters:
        ----------
        rgt: the RGT number of the granule, int
        cycle: the cycle number of the granule, int
        beam: optional, the beam name, str
        data: optional, the ATL10 data, xr.Dataset
        '''
        self.rgt = rgt
        self.cycle = cycle
        if beam is not None: self.beam = beam
        if data is not None: self._data = data
        return
    
    def z_index(self,
                ZBOT: Optional[Union[int,float]] = -100,
                ZTOP: Optional[Union[int,float]] = 14e3,
                ZLOW: Optional[Union[int,float]] = 2e3,
                ZMID: Optional[Union[int,float]] = 6e3,  
               ) -> dict:
        '''
        Generate indice of cloud levels in the vertical grid for 
        high, middle, low, and all clouds uisng the level boundaries. 
        All cloud indices are still needed because the original 
        coordinate start from around -900 m and fold in 15 km range. 
        
        Parameters:
        ----------
        ZBOT: optional, float, the bottom altitude [m] used to subset in the vertical, default -100.
        ZTOP: optional, float, the top altitude [m] used to subset in the vertical, 
              default 14 km. IS2 altitude is folded every 15 km, starting from -900+ m. 
        ZLOW: optional, float, the altitude [m] used to define the top of 
              low level clouds, default 2 km. 
        ZMID: optional, float, the altitude [m] used to define the top of 
              middle level clouds, default 6 km. 
              
        Return:
        ------
        izdict: dictionary containing the indices of zz for different levels. 
        '''
        zz = self._data.ds_va_bin_h.values
        izdict = ATL09._z_index(zz,ZTOP=ZTOP,ZMID=ZMID,ZLOW=ZLOW,ZBOT=ZBOT) 
        return izdict
    
    
    @staticmethod
    def _z_index(zz: np.ndarray,
                 ZBOT: Optional[Union[int,float]] = -100,
                 ZTOP: Optional[Union[int,float]] = 14e3,
                 ZLOW: Optional[Union[int,float]] = 2e3,
                 ZMID: Optional[Union[int,float]] = 6e3,                 
                 ) -> dict:
        '''
        Generate indice of cloud levels in the vertical grid for 
        high, middle, low, and all clouds uisng the level boundaries. 
        All cloud indices are still needed because the original 
        coordinate start from around -900 m and fold in 15 km range. 
        
        Parameters:
        ----------
        zz: np.ndarray, the vertical coordinate used to generate the indices. 
        ZBOT: optional, float, the bottom altitude [m] used to subset in the vertical, default -100.
        ZTOP: optional, float, the top altitude [m] used to subset in the vertical, 
              default 14 km. IS2 altitude is folded every 15 km, starting from -900+ m. 
        ZLOW: optional, float, the altitude [m] used to define the top of 
              low level clouds, default 2 km. 
        ZMID: optional, float, the altitude [m] used to define the top of 
              middle level clouds, default 6 km. 
              
        Return:
        ------
        izdict: dictionary containing the indices of zz for different levels. 
        '''
        izdict = dict()
        izdict['all'] = np.where(np.logical_and(zz>=ZBOT,zz<=ZTOP))[0]
        izdict['low'] = np.where(np.logical_and(zz>=ZBOT,zz<=ZLOW))[0]
        izdict['mid'] = np.where(np.logical_and(zz>=ZLOW,zz<=ZMID))[0]
        izdict['high'] = np.where(np.logical_and(zz>=ZMID,zz<=ZTOP))[0]    
        return izdict
    
    @staticmethod
    def cloudmask(ds, 
                  i0: Optional[InDex] = None, 
                  i1: Optional[InDex] = None, 
                  idx_in: Optional[InDexList]  = None, 
                  mean_cm: Optional[bool] = False,
                  plume: Optional[bool] = False, 
                  shade: Optional[bool] = False, 
                  izdict: Optional[dict] = None,
                  ZBOT: Optional[Union[int,float]] = -100,
                  ZTOP: Optional[Union[int,float]] = 14e3,
                  ZLOW: Optional[Union[int,float]] = 2e3,
                  ZMID: Optional[Union[int,float]] = 6e3,
                  ZBOT_PLUME: Optional[Union[int,float]] = 5e2,
                  topo: Optional[Union[int,np.int64]] = 0
                 ):
        '''
        Generate cloud mask (or mean vertical profile), shaded cloud fraction at
        different cloud levels, and plume cloud mask (or mean vertical profile).
        The input ATL09 xr.Datast can be subsetted before cloud masking using: 
        (1) if idx_in is provided, this list of indice are used;
        (2) elif i0/i1 is provided, they are used;
        (3) else use the complete ds indices
        to generate cloud mask.         
        
        Parameters:
        ----------
        ds: ATL09 data in xr.Dataset used to generate the cloud mask
        i0: optional, the start index in ds to generate the cloud mask, float or int, default None. 
        i1: optional, the end index in ds to generate the cloud mask, float or int, default None. 
        index_in: optional, a list of 1d array of indices in ds used to generate the cloud mask, default None.  
        mean_cm: optional, boolean flag to return mean cloud fraction profiles and statistics, default False. 
        plume: optional, boolean flag to return cloud mask (and shaded cloud fraction) for plume
        shade: optional, boolean flag to return shaded cloud fraction at different levels, default False. 
        izdict: optional, a dictionary containing the indices of vertical coordinate for different
                cloud levels, ['low', 'middle', 'high', 'all'], default None. 
        ZBOT: optional, float, the bottom altitude [m] used to subset in the vertical, default -100.
        ZTOP: optional, float, the top altitude [m] used to subset in the vertical, 
              default 14 km. IS2 altitude is folded every 15 km, starting from -900+ m. 
        ZLOW: optional, float, the altitude [m] used to define the top of 
              low level clouds, default 2 km. 
        ZMID: optional, float, the altitude [m] used to define the top of 
              middle level clouds, default 6 km. 
        ZBOT_PLUME: optional, float, the maximum plume base altitude [m], default 500 m. 
        topo: optional, int, the flag to switch from ATLAS height to AGL height about surface/dem height,
              default 0, which use IS2 surface height when available and dem_h otherwise. If set to negative,
              0 m surface height will be used. 
        Return:
        ------
        cmdict: dictionary containing:
                'CM': (mean) cloud mask vertical profiles; 
                'cld_'+LVL: shaded cloud fraction at different cloud levels;
                'CM_plume': (mean) plume cloud mask vertical profiles; 
                'cld_plume': shaded plume cloud fraction;
                'nprof': number of ATL09 profiles used in the (mean) cloud mask. 
        '''
        
        if isinstance(ZBOT_PLUME,(int,float,np.int64)):
            zbot_plm = np.array([ZBOT_PLUME])
        elif isinstance(ZBOT_PLUME,(list,np.ndarray)):
            zbot_plm = ZBOT_PLUME
        nplm = len(zbot_plm)
        
        zz = ds.ds_va_bin_h.values
        nz = len(zz)
        if izdict is None:  
            izdict = ATL09._z_index(zz,ZBOT=ZBOT,ZTOP=ZTOP,ZMID=ZMID,ZLOW=ZLOW)   
            
        if idx_in is None:
            if i0 is None: i0 = 0
            if i1 is None: i1 = len(ds.delta_time)-1
            idx_in = list(range(i0,i1+1))
            
        nprof = len(idx_in)
        shp = (nz,) if mean_cm else (nprof,nz)
        cmdict = dict()        
        cmdict['nprof'] = nprof
        cmdict['CM'] = np.zeros(shp)
        cmdict['CM_plume'] = np.zeros(shp+(nplm,))
        lvl_list = ['all','low','mid','high'] 
        cpp_list = ['cth','cbh','cdp']
        for cpp in cpp_list:
            cmdict[cpp] = 0 if mean_cm else np.zeros(nprof)
            cmdict[cpp+'_plume'] = np.zeros(nplm) if mean_cm else np.zeros((nprof,nplm))
        for lvl in lvl_list:
            cmdict['cld_'+lvl] = 0 if mean_cm else np.zeros(nprof)
        cmdict['cld_plume'] = np.zeros(nplm) if mean_cm else np.zeros((nprof,nplm))
            
        for ii in range(nprof):
            i_prof = idx_in[ii]
            z0 = 0
            if topo>=0:
                z00 = ds.dem_h[i_prof].values
                z01 = ds.surface_height[i_prof].values
                z0 = z00
                if topo==0 and z01<1e4:
                    z0 = z01
            ztop = ds.layer_top[i_prof].values - z0
            zbot = ds.layer_bot[i_prof].values - z0
                    
            # --- initialize temp data
            cm_i = np.zeros(nz)
            cm_plm_i = np.zeros((nz,nplm))
            cpp_dict = dict()
            cpp_plm_dict = dict()
            for cpp in cpp_list:
                cpp_dict[cpp] = 0
                cpp_plm_dict[cpp] = np.zeros(nplm)
                
            fdx = ztop<ZTOP
            if any(fdx):
                ilays = np.where(fdx)[0]
                cpp_dict['cth'] = ztop[ilays[0]]
                cpp_dict['cbh'] = ds.layer_bot[i_prof].values[[ilays[-1]]]
                cpp_dict['cdp'] = cpp_dict['cth'] - cpp_dict['cbh']
                for i_lay in ilays:
                    ztop_i = ztop[i_lay] 
                    zbot_i = zbot[i_lay]
                    iz = np.where( np.logical_and(zz>=zbot_i,zz<=ztop_i) )[0].astype(int)
                    cm_i[iz] = 1   
                    if plume and i_lay==ilays[-1]:
                        for iplm in range(nplm):
                            if zbot_i<=zbot_plm[iplm]:
                                cm_plm_i[iz,iplm] = 1
                                cpp_plm_dict['cth'][iplm] = ztop_i
                                cpp_plm_dict['cbh'][iplm] = zbot_i
                                cpp_plm_dict['cdp'][iplm] = ztop_i - zbot_i  
                                
                if mean_cm:
                    cmdict['CM'] += cm_i
                    cmdict['CM_plume'] += cm_plm_i
                else:
                    cmdict['CM'][ii] = cm_i
                    cmdict['CM_plume'][ii] = cm_plm_i
                # --- use izdict to find the vertical level indices for shaded CF.
                for lvl in lvl_list+['plume']:
                    dat = cm_plm_i if lvl=='plume' else cm_i[izdict[lvl]]
                    if mean_cm:
                        cmdict['cld_'+lvl] += dat.max(axis=0)
                    else:
                        cmdict['cld_'+lvl][ii] = dat.max(axis=0)
                    
                for cpp in cpp_list:
                    if mean_cm:
                        cmdict[cpp] += cpp_dict[cpp]
                        cmdict[cpp+'_plume'] += cpp_plm_dict[cpp]
                    else:
                        cmdict[cpp][ii] = cpp_dict[cpp]
                        cmdict[cpp+'_plume'][ii] = cpp_plm_dict[cpp]

        if mean_cm:
            for cpp in cpp_list:
                cmdict[cpp] /= cmdict['nprof']
                cmdict[cpp+'_plume'] /= cmdict['nprof']
            for lvl in lvl_list:
                cmdict['cld_'+lvl] /= cmdict['nprof']
        
            
        return cmdict
    
    
class ATL07():
    def __init__(self,
                 rgt:IntType = None,
                 cycle:IntType = None,
                 beam: Optional[str] = None,
                 data: Optional[xr.Dataset] = None) -> None:
        '''
        Initialie the ATL10 instance.
        
        Parameters:
        ----------
        rgt: the RGT number of the granule, int
        cycle: the cycle number of the granule, int
        beam: optional, the beam name, str
        data: optional, the ATL10 data, xr.Dataset
        '''
        self.rgt = rgt
        self.cycle = cycle
        if beam is not None: self.beam = beam
        if data is not None: self._data = data#.to_dataframe().reset_index()
        
        return
    
    def plume_leads(self,plume,dT=.1) -> pd.DataFrame:
        '''
        Find collocating leads given a plume. 
        
        Parameters:
        ----------
        plume: the identified plume in ATL09, pandas.DataFrame
        dT: optional, the additional time allowed to find lead.
            default 0.1s, or 700m. 
            The 0.04s correspondes to 280m in ATL09 profile spacing. 
            
        Return:
        ------
        The collocated leads, pandas.DataFrame
        '''
        t0 = plume.iloc[0].t0
        t1 = plume.iloc[0].t1
        tleads = self.leads
        tdf = tleads[(tleads.t_end>=t0-dT)&(tleads.t_start<=t1+dT)]
        
        return tdf    
    

    def check_leads(self,lead_flag=None) -> None:
        '''
        Check if there is any lead found in the granule. 
        '''
        if lead_flag is None: lead_flag = [1]
        if hasattr(self,'_df'):
            tdata = self._df
        else:
            tdata = self._data.to_dataframe().reset_index()
        zflag = tdata.height_segment_ssh_flag
        contain_leads = any(zflag.isin(lead_flag))
        self.contain_leads = contain_leads
        return 


    def connect_feature(self,feat_flag=None,gap_div=True) -> None:
        '''
        return start and end of height segment ids for continuous leads in granule
        '''
        if feat_flag is None: feat_flag = [1]
        if hasattr(self,'_df'):
            tdata = self._df
        else:
            tdata = self._data.to_dataframe().reset_index()
            self._df = tdata
        zflag = tdata.height_segment_ssh_flag
        stype = tdata.height_segment_type

        # --- find lead segments:
        #     (1) using ssh_flag of 1 for ATL07
        #     (2) using [1,2] or [2] for ATL10
        dsleads = tdata[zflag.isin(feat_flag)]

        seg_id = dsleads.height_segment_id.values
        i0s_in, i1s_in = index_connect(seg_id)
        if gap_div:
            i0s, i1s = ATL07.separate_feature(dsleads,i0s_in,i1s_in)
        else:
            i0s = i0s_in
            i1s = i1s_in
        return seg_id[i0s].astype(int), seg_id[i1s].astype(int)
        
    @staticmethod
    def separate_feature(data: xr.Dataset,
                         i0s_in: InDexList,
                         i1s_in: InDexList,
                         len_crit: Optional[IntType] = 5,
                        ) -> Tuple[InDexList,InDexList]:
        '''
        Separate ATL07/10 feature condidates using gaps in geoseg_end and geoseg_beg
        of nearby height_segments. 
        
        Parameters:
        ----------
        data: ATL07/10 data in xr.Dataset
        i0s_in: a list or array of start indices of the features
        i1s_in: a list or array of end indices of the features
        len_crit: optional, the critical difference of the gap between geoseg_end
                  and geoseg_beg of nearby height segments, default 5. 
        
        Return:
        ------
        i0s: a list or array of start indices of the separated features
        i1s: a list or array of end indices of the separated features
        '''
        
        nfeat_in = len(i0s_in)
        i0s = []
        i1s = []
        i0 = 0
        for i_feat in range(nfeat_in):
            i0 = i0s_in[i_feat]
            i0s.append(i0)
            idx = range(i0s_in[i_feat], i1s_in[i_feat]+1)
            geo_i0 = data.geoseg_beg.values[idx]
            geo_i1 = data.geoseg_end.values[idx]
            geo_di = geo_i0[1:]-geo_i1[:-1]
            fdx = geo_di>len_crit
            if any(fdx):
                ii = np.where(fdx)[0].astype(int)
                for i_seg in ii:
                    i1 = idx[i_seg]
                    i1s.append(i1)
                    i0 = idx[i_seg+1]
                    i0s.append(i0)
            i1 = i1s_in[i_feat]
            i1s.append(i1)
        return i0s, i1s 
    
    @staticmethod
    def locate_gap(data: xr.Dataset,
                   len_crit: Optional[IntType] = 5
                  ) -> pd.DataFrame:
        '''
        Find the gaps in height segment features. 
        '''
        geo_i0 = data.geoseg_beg.values
        geo_i1 = data.geoseg_end.values
        geo_di = geo_i0[1:]-geo_i1[:-1]
        fdx = geo_di>len_crit
        if any(fdx):
            i0s = np.where(fdx)[0].astype(int)
            i1s = i0s + 1 
            
            df = pd.DataFrame(dict(
                t_start=data.delta_time.values[i0s],
                t_end=data.delta_time.values[i1s],
                lat_start=data.latitude.values[i0s], 
                lat_end=data.latitude.values[i1s],
                lon_start=data.longitude.values[i0s], 
                lon_end=data.longitude.values[i1s],
                width=data.seg_dist_x.values[i1s] - data.seg_dist_x.values[i0s],
                sid=data.height_segment_id.values[i0s],
                eid=data.height_segment_id.values[i1s],
                feat='gap'
                #i0 = i0s,
                #i1 = i1s
            ))
        else:
            df = pd.DataFrame()
        return df
    
    def _seglen(self,direct: Optional[bool] = True) -> None:
        '''
        Add height_segment length to another variable in the data. 
        In V003, the length_seg might not be correctly calculated due to 
        the marking of first and last photons in the segments.
        In future release, the direct method should be used or this function 
        should be get rid of. 
        For now, keep the option open to correct the issue with a indirect method. 
        '''
        if direct:
            self._data['seglen'] = self._data.height_segment_length_seg
        else:
            pass
            
        return
    
    def _add_leads(self,wmin=100,lead_flag=None) -> None:
        '''
        Add connected leads to the ATL07 instance. 
        '''
        if lead_flag is None: lead_flag = [1]
        if not hasattr(self,'contain_leads'):
            self.check_leads(lead_flag=lead_flag)  
            
        if 'seglen' not in self._data.keys(): self._seglen()
        if not hasattr(self,'_df'): 
            self._df = self._data.to_dataframe().reset_index()
        data = self._df
        
        # --- if no leads, set leads to empty. 
        if self.contain_leads:
            if not hasattr(self ,'_leads'): 
                sid, eid = self.connect_feature(feat_flag=lead_flag)
            
            nlead = len(sid)
            self.nleads = nlead
            self._leads = []
            self._lead_sid = sid
            self._lead_eit = eid
            
            for i_lead in range(nlead):    
                # --- pandas.DataFrame is more efficient than xr.Dataset in data subsetting
                #     likely due to the dimension matching of the xr.where
                t_lead = Lead(data, sid[i_lead], eid[i_lead])
                self._leads.append(t_lead)
        else:
            self._leads = []
    
        return
    
    def _add_floes(self,wmin=100,floe_flag=None) -> None:
        '''
        Add connected leads to the ATL07 instance. 
        '''
        if floe_flag is None: floe_flag = [0]
        if 'seglen' not in self._data.keys(): self._seglen()
        if not hasattr(self,'_floes'): sid, eid = self.connect_feature(feat_flag=floe_flag)
        
        nfloe = len(sid)
        self.nfloes = nfloe
        self._floes = []
        self._floe_sid = sid
        self._floe_eit = eid    
        
        for i_floe in range(nfloe):  
            # --- pandas.DataFrame is more efficient than xr.Dataset in data subsetting
            #     likely due to the dimension matching of the xr.where        
            if not hasattr(self,'_df'): 
                self._df = self._data.to_dataframe().reset_index()
            data = self._df
            t_floe = Floe(data, sid[i_floe], eid[i_floe])
            self._floes.append(t_floe)
    
        return
        
    @property
    def floes(self):
        if not hasattr(self,'_floes'): self._add_floes()
        if isinstance(self._floes,pd.DataFrame):
            odf = self._floes
        else:
            odf = pd.DataFrame( [x.__dict__ for x in self._floes] )
        return odf
        
    @property
    def leads(self):
        if not hasattr(self,'_leads'): self._add_leads()
        if len(self._leads)>0:
            if isinstance(self._leads,pd.DataFrame):
                odf = self._leads
            else:
                odf = pd.DataFrame( [x.__dict__ for x in self._leads] )
        else:
            odf = pd.DataFrame()
        return odf
    
def index_connect(idx):
    nids = len(idx)
    id_incr = np.diff(idx)
    fdx = id_incr>1
    if any(fdx):
        i_skip = np.where(fdx)[0]

        nskip = len(i_skip)
        i0s = np.zeros(nskip+1)
        i1s = np.zeros(nskip+1)
        for k in range(nskip):
            k_jump = i_skip[k]
            if k==0:
                i0s[k] = 0
                i1s[k] = k_jump
            else:
                i0s[k] = i1s[k-1]+1
                i1s[k] = k_jump
        i0s[k+1] = i1s[k]+1
        i1s[k+1] = nids-1
        i0s = i0s.astype(int)
        i1s = i1s.astype(int)
    else:
        # --- only one type of feature, likely all floe
        i0s = np.array([0]).astype(int)
        i1s = np.array([nids-1]).astype(int)
    return i0s,i1s#np.array(idx)[i0s].astype(int), np.array(idx)[i1s].astype(int)
