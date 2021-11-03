import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Union, Optional, List, NoReturn#, Type

from .utils import strong_beam


    
class is2db:
    '''
    The is2db object contains:
    
    Attributes:
    ----------
    prdlist: a list of IS2 ATL products
    ATL07: optional, a pandas.DataFrame of information for ATL07 data, 
           including file name (fn), rgt, cycle, time, 
           and optional version and release number.
    ATL09: optional, a pandas.DataFrame of information for ATL09 data, 
    ATL10: optional, a pandas.DataFrame of information for ATL10 data, 
    
    Methods:
    -------
    add_product: add product database from data root directory, product name, and subset tag. 
    query      : subset the databset using rgt, cycle, t_start, and t_end
    update_db  : update the database for a specific product    
    build_db   : staticmethod, build the database for a product, with optional subset tag. 
    disp_gran  : display the file names of IS2 products for a granule specified by rgt and cycle. 
    info       : print a summary of data products in this database.     
        
    Examples:
    -------
    >>> is2sub = is2db(dataroot=Path('/media/Data4T/IS2/V003'),prdlist=['ATL07','ATL09','ATL10'])
    Another approach: create an empty instance and update the products later. 
    >>> is2sub = is2db()
    >>> df = is2db.build_db(DataRoot,'ATL07',subset='rsub')
    >>> is2sub.update_db('ATL07',df)
    '''
    def __init__(self,
                 dataroot: Optional[ Union[str,Path] ] = None,
                 prdlist: Optional[ List[str] ] = None, 
                 vrl: Optional[bool] = True,
                 strong: Optional[bool] = True,
                 filename = None,
                ) -> NoReturn:
        '''
        Initialize ICESat-2 database class object. 
        This requires the IS2 data to be stored in the following way. 
        Dataroot/prd/[subset]/
        For now, data version is included in Dataroot. 
        The data requested with icepyx for the same version can contain different releases. 
        By default, the r beams of ATL07 and ATL10 are stored in subset='rsub' an l beams in lsub. 
        the ATL09 files are stored with subset='subset'. 
        Addition data subset should be stored in a similar fashion and added to the is2db instance
        using add_product. 
        
        Parameters:
        ----------
        dataroot: optional, the root directory of the database, string or pathlib.Path object
        prdlist : optional, the list of IS2 products, subset of ['ATL07','ATL09','ATL10']
        vrl     : optional, the flag to add data version and release number, default True
        strong  : optional, the flag to filter strong beam data only, default True
        '''
        self.prdlist = prdlist if prdlist is not None else []
        if dataroot is not None: 
            self.root = dataroot
        if filename is None:
            for prd in self.prdlist:
                subset = 'rsub' if prd!='ATL09' else 'subset'
                self.add_product(prd,subset=subset,vrl=vrl,strong=strong)
                if prd!='ATL09':
                    self.add_product(prd,subset='lsub',vrl=vrl,strong=strong)
        else:
            self.from_file(filename)
        return
    
    
    def from_file(self,filename: Union[str,Path]) -> NoReturn:
        '''
        read product data lists from file. 
        
        Parameters:
        -----------
        filename: the name containing the list of products and the csv files
                  of the data lists for that product.
        '''
        with open(filename) as fid:
            df = pd.read_csv(filename)
            if len(df)>0:
                self.prdlist = df.prd.values
                for i in range(len(df)):
                    prd = df.iloc[i].prd
                    fn  = df.iloc[i].filename
                    prddf = pd.read_csv(fn)
                    setattr(self,prd,prddf)
            else:
                raise ValueError('No data added yet.')
            
        return #df
    
    def to_file(self,
                filename: Union[str,Path],
                verbose: Optional[bool] = True,
               ) -> NoReturn:
        '''
        Write is2db instance to files.
        The list of products and the filenames of the csv file that
        contais the list of data in a given product is written to a 
        index file specfied by filename. 
        
        Parameters:
        ----------
        filename: the name containing the list of products and the csv files
                  of the data lists for that product.
        verbose: optional, the boolean flag to print out products and the csv file names. 
        '''
        outfn = Path(filename)
        outdir = outfn.parent/'.is2db'/outfn.stem
        if hasattr(self,'prdlist'):
            if not outdir.exists(): outdir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError('No product added in granule.')
        with open(outfn,'w') as fid:
            fid.write('prd,filename\n')    
            for prd in self.prdlist:                    
                fn = outdir/(prd+'.csv')
                if verbose: print(prd,fn)
                getattr(self,prd).to_csv(fn,index=False)    
                fid.write(prd+','+str(fn)+'\n')
        return
    
    def add_product(self,
                    prd: str ,  
                    subset: str = None,
                    dataroot: Optional[ Union[str,Path] ] = None,
                    vrl: Optional[bool] = True,
                    strong: Optional[bool] = True,
                   ) -> NoReturn :
        '''
        Add information of the IS2 data product to the database. 
        
        Parameters:
        ----------
        prd     : IS2 product name, str, choose from ['ATL07','ATL09','ATL10']
        subset  : the subset directory of the IS2 data, str. 
        dataroot: optional, the root directory of the database, string or pathlib.Path object
        vrl     : optional, the flag to add data version and release number, default True
        strong  : optional, the flag to filter strong beam data only, default True
        '''
        
        if self.root is None and dataroot is None:
            raise ValueError('No dataroot is provided')
        elif dataroot is None:
            dataroot = self.root
        else:
            self.root = dataroot
        
        prdlist = ['ATL07','ATL09','ATL10']
        errmsg = '{0} is not in the supported list: {1}.'.format(prd, prdlist)
        assert prd in prdlist, errmsg       
        
        df = self.build_db(dataroot,prd,subset=subset,vrl=vrl,strong=strong)
        if hasattr(self,prd): 
            tdf = getattr(self,prd)
            df = tdf.append(df,ignore_index=True)
            df.sort_values('time',ignore_index=True,inplace=True)
        setattr(self,prd,df)
        if prd not in self.prdlist:
            self.prdlist.append(prd)
        return
    
    def query(self,
              cycle: Optional[ Union[int, np.int64] ] = None,
              rgt: Optional[ Union[int, np.int64] ] = None,
              t_start: Optional[ pd.Timestamp ] = None,
              t_end: Optional[ pd.Timestamp ] = None) -> 'is2db':
        '''
        Query the database and return a subset instance.
        
        Parameters:
        ----------
        cycle: optional, cycle number of the granule, int.
        rgt: optional, rgt number of the granule, int.
        t_start: optional, the starting time of the query, pandas.Timestamp
        t_end: optional, the end time of the query, pandas.Timestamp
        
        Return:
        ------
        t_db: the subset database, is2db object
        '''
        assert hasattr(self,'prdlist'), 'No prdlist found in is2db object.'
        
        t_db = is2db()
        for prd in self.prdlist:
            tdf = getattr(self,prd)
            if rgt is not None:
                if isinstance(rgt,(int,np.int64)): tdf = tdf[tdf.rgt==rgt]
                if isinstance(rgt,list): tdf = tdf[tdf.rgt.isin(rgt)]
            if cycle is not None:
                if isinstance(cycle,(int,np.int64)): tdf = tdf[tdf.cycle==cycle]
                if isinstance(cycle,list): tdf = tdf[tdf.cycle.isin(cycle)]
            if t_start is not None and t_end is not None:
                tdf = tdf[ (tdf.time>=t_start)&(tdf.time<=t_end) ]
            t_db.update_db(prd=prd,db_df=tdf)
            if hasattr(self,'root'): t_db.update_db(dataroot=self.root)
        return t_db
    
    def update_db(self, 
                  prd: Optional[str] = None, 
                  db_df: Optional[pd.DataFrame] = None,
                  dataroot: Optional[ Union[str,Path] ] = None,
                 ) -> NoReturn :
        '''
        Update the is2db instance using product name and the pandas.DataFrame containing
        the data information.
        
        Parameters:
        ----------
        prd: product name, str.
        db_df: product data information, pd.DataFrame
        
        '''
        prdlist = ['ATL07','ATL09','ATL10']
        errmsg = '{0} is not supported. Please choose from {1}'.format(prd,prdlist)
        if prd is not None and db_df is not None:
            assert prd in prdlist, errmsg
            setattr(self,prd,db_df)
            if prd not in self.prdlist: self.prdlist.append(prd)
        if dataroot is not None: self.root = dataroot
        return
    
    def disp_gran(self,rgt: Union[int,np.int64],cycle: Union[int,np.int64]) -> NoReturn:
        '''
        Print the filenames of all products for this granule. 
        
        Parameters:
        ----------
        rgt: the reference ground track number, int
        cycle: the IS2 cycle number, int
        '''
        print('IS2 times and files for RGT {0}, cycle {1}:'.format(rgt,cycle))
        t_gran = self.query(cycle=cycle,rgt=rgt)
        for prd in self.prdlist:
            tdf = getattr(self,prd)
            print('{0}: {1} :: {2}'.format(prd,tdf.iloc[0].time,tdf.iloc[0].fn))
        return
    
    def info(self) -> NoReturn:
        '''
        Print brief information about database. 
        '''
        if hasattr(self,'root'): 
            print('Dataroot: {0} .'.format(self.root))
        else:
            print('No dataroot added yet.')
        if len(self.prdlist)==0:
            print('No product added to database yet.')
        else:
            print('prdlist: {0} .'.format(self.prdlist))
            dict_list = []
            for prd in self.prdlist:
                # DevNote: add warning for prd not in the list later.
                if hasattr(self,prd):
                    df = getattr(self,prd)
                    if len(df)>0:
                        t_start = df.iloc[0].time
                        t_end = df.iloc[-1].time
                        nfns = len(df)
                        tdict = dict(prd=prd,nfns=nfns,
                                     t_start=t_start,t_end=t_end)
                        dict_list.append(tdict)
            odf = pd.DataFrame(dict_list)
            print(odf)
        return
    
    @staticmethod
    def build_db(datroot: Union[Path,str],
                 prd: str,
                 beam: Optional[int] = None,
                 subset: Optional[str] = None,
                 fn_pre: Optional[str] = None,
                 db_df: Optional[pd.DataFrame] = None,
                 vrl: Optional[bool] = True,
                 strong: Optional[bool] = True,
                 filetype: Optional[str] = 'h5',
                ) -> pd.DataFrame:
        '''
        Add IS2 files and info to a Pandas DataFrame. 
        This function requires data to be organized in the following ways:
        1) datroot/prd/subset/is2.h5 
        2) datroot/prd/is2.h5
        For the first case, the subdirectory to the subset is needed.
        Please provide db_df to update existing db_df with another set of data. 

        Parameters:
        ----------
        datroot: root directory of dataset
        prd: product name for the IS2 data, ATL10, ATL09
        subset: the subdirectory name to the subset data
        db_df (optional): the IS2 database stored in Pandas DataFrame, to be updated
        vrl: boolean flag to add data version and release number to the database
 
        Return:
        df: Pandas DataFrame of the constructed/updated IS2 database 
        '''
        # --- segments for glob pattern match
        prefix = '' if subset is None else 'processed_'
        hemcod = '' if prd=='ATL09' else '-0[1-2]'    
        ymdhms = '20[1-9][0-9][0-1][0-9][0-3][0-9]'+\
                 '[0-2][0-9]'+'[0-6][0-9]'*2
        rgtcs = '[0-1]'+'[0-9]'*5+'0[0-9]'
        vvv = '[0-9]'*3
        rr  = '[0-9]'*2
        pattern = '{0}{1}{2}_{3}_{4}_{5}_{6}.{7}'.format(
            prefix, prd, hemcod, ymdhms,rgtcs,vvv,rr,filetype )
        if fn_pre is not None:
            pattern = fn_pre +'_'+pattern

        # --- re pattern matcher 
        rx = re.compile('(......._)?(processed_)?(ATL\\d{2})(-\\d{2})?_(\\d{4})(\\d{2})(\\d{2})(\\d{2})(\\d{2})'
               '(\\d{2})_(\\d{4})(\\d{2})(\\d{2})_(\\d{3})_(\\d{2})(.*?).(.*?)$')

        if subset is None: subset =  ''
        if beam is None:
            datdir = Path(datroot)/prd/subset
        else:
            datdir = Path(datroot)/prd/str(beam)/subset
        fns = sorted( datdir.glob(pattern) )   
        dict_list = []
        #for fn in fns:
        for ifn,fn in enumerate(fns):
            #print(rx.findall(fn.name).pop())
            FNPRE,PFX,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,VERS,RL,AUX,SFX = \
                rx.findall(fn.name).pop()
            tt = pd.to_datetime(YY+MM+DD+HH+MN+SS,format='%Y%m%d%H%M%S')
            fnd = False
            if strong and prd!='ATL09':
                sbeam = strong_beam(tt)
                if sbeam=='r' and subset in ['rsub']: fnd = True 
                if sbeam=='l' and subset in ['lsub']: fnd = True
            else: fnd = True
            if fnd:    
                tdict = dict(prd=PRD,fn=fn,rgt=int(TRK),cycle=int(CYCL),time=tt)
                if vrl: tdict.update(dict(version=VERS,release=RL))
                dict_list.append(tdict)
        tdf = pd.DataFrame(dict_list).drop_duplicates()
        if db_df is None:
            df = tdf
        else:
            df = db_df.append(tdf,ignore_index=True)
            df.sort_values('time',ignore_index=True,inplace=True)
        
        return df #

