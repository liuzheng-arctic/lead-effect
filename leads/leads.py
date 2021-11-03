import numpy as np
import pandas as pd
import xarray as xr

class Lead:
    def __init__(self,df,sid,eid, feat='lead'):
        assert isinstance(sid,(int,np.int64)), 'sid should be integer'
        assert isinstance(eid,(int,np.int64)), 'eid should be integer'
        dflead = df[df.height_segment_id.isin(range(sid,eid+1))]
        feat_flag = dflead.iloc[0].height_segment_ssh_flag
        if any(abs(dflead.height_segment_ssh_flag.values-feat_flag)!=0):
            errmsg = 'Inconsistent ssh_flag in the height segments {0} to {1}.'.format(sid,eid)
            raise ValueError(errmsg)

        self.lat_start = dflead.latitude.values[0]
        self.lon_start = dflead.longitude.values[0]
        self.lat_end = dflead.latitude.values[-1]
        self.lon_end = dflead.longitude.values[-1]
        self.width = dflead.seglen.values.sum()
        self.sid = sid
        self.eid = eid
        self.t_start = dflead.delta_time.values[0]
        self.t_end = dflead.delta_time.values[-1]
            
        return
    
    def to_df(self):
        return pd.DataFrame(self.__dict__,index=[0])
    
    
class Floe(Lead):
    def __init__(self,df,sid,eid):
        super().__init__(df,sid,eid,feat='floe')
        return
