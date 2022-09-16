'''
La librería contiene las clases y funciones que permiten extraer los datos de cambio climático en la zona de estudio
y realizar la corrección de sesgo mediante el método Scalled Distribution Mapping.
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''


from netCDF4 import Dataset
import numpy as np
import pandas as pd
import tqdm
from osgeo import gdal, ogr, osr
import pandas as pd
import os
import glob
import datetime
from math import floor
from pyproj import Proj, transform
import pickle
import yaml
from SIMPCCe.CORRECCION_SESGO import *
from SIMPCCe.REGRESION import SIM_REGRESION

class Climate_Change(object):
    """
    Con esta clase se realiza todo el análisis de cambio climático.
    
    Datos de Entrada:
    -----------------
    path_project:     string.  Directorio del proyecto
    name_project:     string.  Nombre del proyecto
    path_data:        string.  Directorio donde se encuentran los datos descargados
    nombre_embalse:   string.  Nombre del embalse
    logging:    True o False. True si se quiere visualizar el proceso, False si no se quiere. 

    """
    def __init__ (self,path_project,name_project,path_data,nombre_embalse,logging =True):
        self.path_project    = path_project
        self.path_data       = path_data
        self.nombre_embalse  = nombre_embalse
        if logging == False:
            tqdm.tqdm(disable =True)
            
        self.file_yml = path_project+'/'+name_project+'.yml'
        
    def extrac_climate_change_AEMET(self):
        """
        Con esta función se extraen los datos de cambio climático en los puntos de la cuenca de estudio.
        
        """
    
        self.models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']

        self.puntos_cuenca = pd.read_csv(self.path_project+'/01_CLIMA/Puntos_Cuenca.csv',index_col=0)
        inProj = Proj(init='epsg:25830')
        outProj = Proj(init='epsg:4326')
        
        dictionary_Hist={v: { model: {} for model in self.models} for v in ['Prec','Tmax','Tmin']}
        dictionary_CC={v: {rcp: {model: {} for model in self.models} for rcp in ['rcp45', 'rcp85']} for v in ['Prec','Tmax','Tmin']}

        for sc in ['HIST','RCP45','RCP85']:
            if sc =='HIST':
                scenario = 'historical'
            elif sc =='RCP45':
                scenario = 'rcp45'
            elif sc =='RCP85':
                scenario = 'rcp85'
            for var in ['PRCPTOT','TXMM','TNMM']:
                if var == 'PRCPTOT':   
                    name = 'Prec'
                elif var == 'TXMM':
                    name = 'Tmax'
                elif var == 'TNMM':
                    name = 'Tmin'
                
                for m in self.models:
                    if sc=='HIST':
                        Dataframe = pd.DataFrame(index=pd.date_range(start='1961-01-01',end='2005-12-31',freq='M'),columns=self.puntos_cuenca.index)
                        Dataframe.index = Dataframe.index.date - pd.offsets.MonthBegin(1)
                    else:
                        Dataframe = pd.DataFrame(index=pd.date_range(start='2006-01-01',end='2100-12-31',freq='M'),columns=self.puntos_cuenca.index)
                        Dataframe.index = Dataframe.index.date - pd.offsets.MonthBegin(1)

                    files = glob.glob(self.path_data+'/AEMET/CAMBIO_CLIMATICO/'+var+'/'+sc+'/'+var+'_'+m+'*')

                    if os.path.exists(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/'+name+'_month_CORDEX_'+scenario+'_'+m+'_'+'r1i1p1.csv'):
                        continue
                    else:
                        os.makedirs(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/',exist_ok=True)
                        for i,ii in enumerate(tqdm.tqdm(files)):

                            #flow[:,:,i]=np.flipud(np.loadtxt(path+'acaes'+str(ii.year)+'_'+str(ii.month)+'.asc',skiprows=6))
                            ds = gdal.Open(ii, gdal.GA_ReadOnly)
                            gt   = ds.GetGeoTransform()

                            date_time_obj = datetime.datetime.strptime(ii.split('_')[-1].split('.')[0], '%Y%m')

                            value_month = []

                            xx = self.puntos_cuenca.loc[:,'COORDX'].values
                            yy = self.puntos_cuenca.loc[:,'COORDY'].values

                            mx = np.array(transform(inProj,outProj,xx,yy)[0]).astype(float)
                            my = np.array(transform(inProj,outProj,xx,yy)[1]).astype(float)
                            if gt[0]>0:
                                gt = (gt[0]-360,gt[1],gt[2],gt[3],gt[4],gt[5])
                            try:
                                px = np.floor((mx- gt[0]) / gt[1]).astype(int) #x pixel
                                py = np.floor((my - gt[3]) / gt[5]).astype(int) #y pixel

                                Dataframe.loc[str(date_time_obj.date()),:] = ds.ReadAsArray().T[px,py]
                            except:
                                continue
                        if sc=='HIST':
                            dictionary_Hist[name][m] = Dataframe
                        else:
                             dictionary_CC[name][scenario][m] = Dataframe
                            
                            
                        Dataframe.to_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/'+name+'_month_CORDEX_'+scenario+'_'+m+'_'+'r1i1p1.csv')
                        
        if os.path.exists(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+"dict_hist.pickle")==False:
            pickle_out = open(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+"dict_hist.pickle","wb")
            pickle.dump(dictionary_Hist, pickle_out)
            pickle_out.close()

            pickle_out = open(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+"dict_CC.pickle","wb")
            pickle.dump(dictionary_CC, pickle_out)
            pickle_out.close()
                        
                        
    def correccion_sesgo(self):
        """
        Con esta función se ecorrige el sesgo de las series de cambio climático.
        
        """

        for var in ['Prec','Tmax','Tmin']:
            if os.path.exists(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+var+'/')==False:
                os.makedirs(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+var+'/')
            if var == 'Prec':
                variable = 'Precipitacion'
            elif var=='Tmax':
                variable = 'Temperatura_Maxima'
            elif var=='Tmin':
                variable = 'Temperatura_Minima'
            for m in self.models:
                print('Corrigiendo serie histórica de la variable '+var+' del modelo '+m)
                Serie_hist = pd.read_csv(self.path_project+'/01_CLIMA/'+variable+'.csv',index_col=0,parse_dates=True)
                Serie_hist = Serie_hist.interpolate(method='linear', axis=1).ffill().bfill()
                Serie_hist.index = Serie_hist.index.date - pd.offsets.MonthBegin(1)
                serie_raw  = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+var+'/'+var+'_month_CORDEX_historical_'+m+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
                
                Serie_hist_correc     = pd.DataFrame(index=pd.date_range(start='1976-01-01', end='2005-12-31', freq='M'),columns = self.puntos_cuenca.index)
                Serie_hist_correc.index = Serie_hist_correc.index.date - pd.offsets.MonthBegin(1)
                
                
                for c in self.puntos_cuenca.index:
                    serie_hist_time = serie_raw.loc['1976':'2005',str(c)].dropna()
                
                    for month in range(1,13):

                        Serie_concat = pd.concat((Serie_hist.loc[Serie_hist.index.month==month,str(c)],serie_raw.loc[serie_raw.index.month==month,str(c)]),axis=1).dropna()

                        Bias_correction   =  bias_correction(Serie_concat.iloc[:,0].values.flatten().astype(float), 
                                                                            Serie_concat.iloc[:,1].values.flatten().astype(float), 
                                                                            serie_hist_time.loc[serie_hist_time.index.month==month].values.flatten().astype(float))

                        index_time = serie_hist_time.loc[serie_hist_time.index.month==month].index

                        if var == 'Prec':
                                variable_2 = 'precipitation'
                        else:
                            variable_2 = 'temperature'

                        Serie_hist_correc.loc[index_time,c]  = Bias_correction.scaled_distribution_mapping(variable_2)
                        
                Serie_hist_correc.to_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+var+'/'+var+'_month_CORDEX_'+'historical'+'_'+m+'_'+'r1i1p1.csv')

                for rcp in ['rcp45','rcp85']:
                    print('Corrigiendo serie del escenario '+rcp+' de la variable '+var+' para el modelo '+m)
                    
                    if os.path.exists(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+var+'/'+var+'_month_CORDEX_'+rcp+'_'+m+'_'+'r1i1p1.csv'):
                        continue
                    else:
                    
                        Serie_CC = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+var+'/'+var+'_month_CORDEX_'+rcp+'_'+m+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)

                        Serie_CC_correc     = pd.DataFrame(index=pd.date_range(start='2006-01-01', end='2100-12-31', freq='M'),columns = self.puntos_cuenca.index)
                        Serie_CC_correc.index = Serie_CC_correc.index.date - pd.offsets.MonthBegin(1)

                        for c in self.puntos_cuenca.index:
                            
                            for month in range(1,13):

                                Serie_concat = pd.concat((Serie_hist.loc[Serie_hist.index.month==month,str(c)],serie_raw.loc[serie_raw.index.month==month,str(c)]),axis=1).dropna()

                                for p in ['2011_2040','2041_2070','2071_2100']:
                                    year_ini = p.split('_')[0]
                                    year_fin = p.split('_')[1]

                                    Serie_CC_time = Serie_CC.loc[year_ini:year_fin,str(c)].dropna()

                                    Bias_correction   =  bias_correction(Serie_concat.iloc[:,0].values.flatten().astype(float), 
                                                                    Serie_concat.iloc[:,1].values.flatten().astype(float), 
                                                                    Serie_CC_time.loc[Serie_CC_time.index.month==month].values.flatten().astype(float))
                                    
                                    index_time = Serie_CC_time.loc[Serie_CC_time.index.month==month].index

                                    if var == 'Prec':
                                        variable_2 = 'precipitation'
                                    else:
                                        variable_2 = 'temperature'

                                    Serie_CC_correc.loc[index_time,c]  = Bias_correction.scaled_distribution_mapping(variable_2)

                        Serie_CC_correc.to_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+var+'/'+var+'_month_CORDEX_'+rcp+'_'+m+'_'+'r1i1p1.csv')
                        
                        
    def ejec_aport_cc_(self):
        """
        Con esta función se simulan las aportaciones correspondientes a cada modelo y escenario de cambio climático.
        
        """
        models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
        
        file_model_Reg = self.path_project+'/04_REGRESION/'+self.nombre_embalse+'_'+'ANN'+'.sav'
        file_model_PCA = self.path_project+'/04_REGRESION/'+self.nombre_embalse+'_'+'PCA'+'.sav'
        
        
        
        for nmodel in tqdm.tqdm(models):
            
            print('### Ejecutando simulación de aportaciones modelo '+nmodel+' ###')
            
            PREC_HIST = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Prec'+'/'+'Prec'+'_month_CORDEX_'+'historical'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMAX_HIST = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmax'+'/'+'Tmax'+'_month_CORDEX_'+'historical'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMIN_HIST = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmin'+'/'+'Tmin'+'_month_CORDEX_'+'historical'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            
            
            PREC_RCP45 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Prec'+'/'+'Prec'+'_month_CORDEX_'+'rcp45'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMAX_RCP45 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmax'+'/'+'Tmax'+'_month_CORDEX_'+'rcp45'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMIN_RCP45 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmin'+'/'+'Tmin'+'_month_CORDEX_'+'rcp45'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)

            PREC_RCP85 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Prec'+'/'+'Prec'+'_month_CORDEX_'+'rcp85'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMAX_RCP85 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmax'+'/'+'Tmax'+'_month_CORDEX_'+'rcp85'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)
            TMIN_RCP85 = pd.read_csv(self.path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'+'Tmin'+'/'+'Tmin'+'_month_CORDEX_'+'rcp85'+'_'+nmodel+'_'+'r1i1p1.csv',index_col=0,parse_dates=True)

            
            SIM_HIST = SIM_REGRESION(file_model_Reg,file_model_PCA,
                                          PREC_HIST,
                                          TMAX_HIST,
                                          TMIN_HIST) 
            SIM_HIST.simulation()
            SIM_HIST.save_series('Aportaciones.'+nmodel+'.HISTORICAL',self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')
            
            
            SIM_RCP45 = SIM_REGRESION(file_model_Reg,file_model_PCA,
                                           PREC_RCP45,
                                           TMAX_RCP45,
                                           TMIN_RCP45) 
            SIM_RCP45.simulation()
            SIM_RCP45.save_series('Aportaciones.'+nmodel+'.RCP_45',self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')

            SIM_RCP85 = SIM_REGRESION(file_model_Reg,file_model_PCA,
                                           PREC_RCP85,
                                           TMAX_RCP85,
                                           TMIN_RCP85) 
            SIM_RCP85.simulation()
            SIM_RCP85.save_series('Aportaciones.'+nmodel+'.RCP_85',self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')
            
            with open(self.file_yml) as file:
                params = yaml.load(file, Loader=yaml.FullLoader)

            params['ejec_aport_cc']   = 'Si'

            with open(self.file_yml, 'w') as file:
                documents = yaml.dump(params, file)