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
from pyproj import Transformer
import xarray as xr
import sys

def convertir_a_datetime64(ds):
    """
    Convierte el eje 'time' de un Dataset a datetime64[ns] usando pandas,
    y elimina todas las fechas que no puedan convertirse (NaT).

    Parámetros:
    -----------
    ds : xarray.Dataset

    Retorno:
    --------
    ds : xarray.Dataset con time en datetime64[ns] y sin entradas inválidas
    """
    import pandas as pd
    import numpy as np

    try:
        time_vals = ds['time'].values

        # Si ya es datetime64
        if np.issubdtype(time_vals.dtype, np.datetime64):
            #print("✅ El eje 'time' ya está en datetime64[ns].")
            return ds

        # Convertir a string y luego a datetime64, eliminando las no convertibles
        converted_time = pd.to_datetime([str(t) for t in time_vals], errors='coerce')

        # Filtrar NaT
        valid_mask = ~pd.isna(converted_time)
        n_invalid = (~valid_mask).sum()

        # if n_invalid > 0:
        #     print(f"⚠️ Eliminadas {n_invalid} fechas inválidas (NaT).")

        # Aplicar la máscara para filtrar
        ds = ds.isel(time=valid_mask)
        ds = ds.assign_coords(time=converted_time[valid_mask])

        #print("✅ Eje temporal convertido a datetime64[ns] y depurado.")
        return ds

    except Exception as e:
        print(f"❌ Error inesperado al convertir el eje 'time': {e}")
        return ds
    


def extrac_climate_change_AEMET(path_project,path_data):
        """
        Con esta función se extraen los datos de cambio climático en los puntos de la cuenca de estudio.
        
        """
    
        models = ['CLMcom-CCLM4-8-17|CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17|MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17|MPI-M-MPI-ESM-LR','KNMI-RACMO22E|ICHEC-EC-EARTH', 
              'KNMI-RACMO22E|MOHC-HadGEM2-ES','MPI-CSC-REMO2009|MPI-M-MPI-ESM-LR',
              'SMHI-RCA4|CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4|IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4|MOHC-HadGEM2-ES', 'SMHI-RCA4|MPI-M-MPI-ESM-LR']

        puntos_cuenca = pd.read_csv(path_project+'/01_CLIMA/Puntos_Cuenca.csv',index_col=0)
        inProj = Proj(init='epsg:25830')
        outProj = Proj(init='epsg:4326')
        
        dictionary_Hist={v: { model: {} for model in models} for v in ['Prec','Tmax','Tmin']}
        dictionary_CC={v: {rcp: {model: {} for model in models} for rcp in ['rcp45', 'rcp85']} for v in ['Prec','Tmax','Tmin']}

        for sc in ['HIST','RCP45','RCP85']:
            if sc =='HIST':
                scenario = 'historical'
            elif sc =='RCP45':
                scenario = 'rcp45'
            elif sc =='RCP85':
                scenario = 'rcp85'
            for var in ['PRCPTOT','TXMM','TNMM']:
                if var == 'PRCPTOT':   
                    name = 'pr'
                elif var == 'TXMM':
                    name = 'tasmax'
                elif var == 'TNMM':
                    name = 'tasmin'
                
                for m in models:
                    if sc=='HIST':
                        Dataframe = pd.DataFrame(index=pd.date_range(start='1961-01-01',end='2005-12-31',freq='M'),columns=puntos_cuenca.index)
                        Dataframe.index = Dataframe.index.date - pd.offsets.MonthBegin(1)
                    else:
                        Dataframe = pd.DataFrame(index=pd.date_range(start='2006-01-01',end='2100-12-31',freq='M'),columns=puntos_cuenca.index)
                        Dataframe.index = Dataframe.index.date - pd.offsets.MonthBegin(1)

                    #files = glob.glob(self.path_data+'/AEMET/CAMBIO_CLIMATICO/'+var+'/'+sc+'/'+var+'_'+m+'*')
                    files = glob.glob(path_data+var+'/'+sc+'/'+var+'_'+m.replace('|', '_')+'*')

                    if os.path.exists(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/'+name+'_month_CORDEX_'+scenario+'_'+m+'_'+'r1i1p1.csv'):
                        Dataframe = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/'+name+'_month_CORDEX_'+scenario+'_'+m+'_'+'r1i1p1.csv',
                                                index_col=0, parse_dates=True)
                    else:
                        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/',exist_ok=True)
                        for i,ii in enumerate(tqdm.tqdm(files)):

                            #flow[:,:,i]=np.flipud(np.loadtxt(path+'acaes'+str(ii.year)+'_'+str(ii.month)+'.asc',skiprows=6))
                            ds = gdal.Open(ii, gdal.GA_ReadOnly)
                            gt   = ds.GetGeoTransform()

                            date_time_obj = datetime.datetime.strptime(ii.split('_')[-1].split('.')[0], '%Y%m')

                            value_month = []

                            xx = puntos_cuenca.loc[:,'COORDX'].values
                            yy = puntos_cuenca.loc[:,'COORDY'].values

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
                        Dataframe.to_csv(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+name+'/'+name+'_month_CORDEX_'+scenario+'_'+m+'_'+'r1i1p1.csv')
                    if sc=='HIST':
                        dictionary_Hist[name][m] = Dataframe
                    else:
                         dictionary_CC[name][scenario][m] = Dataframe
                            
        pickle_out = open(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+"dict_hist.pickle","wb")
        pickle.dump(dictionary_Hist, pickle_out)
        pickle_out.close()

        pickle_out = open(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/'+"dict_CC.pickle","wb")
        pickle.dump(dictionary_CC, pickle_out)
        pickle_out.close()

        return models

def extract_climate_change_CMIP6(path_project,path_data):
        """
        Extrae los datos de cambio climático en los puntos de la cuenca desde ficheros NetCDF.
        """

        models = ['CNRM-ESM2-1_r1i1p1f2_gr',
                    'KACE-1-0-G_r1i1p1f1_gr',
                    'IPSL-CM6A-LR_r1i1p1f1_gr',
                    'CanESM5_r1i1p1f1_gn',
                    'ACCESS-ESM1-5_r1i1p1f1_gn',
                    'NESM3_r1i1p1f1_gn',
                    'MPI-ESM1-2-HR_r1i1p1f1_gn',
                    'INM-CM4-8_r1i1p1f1_gr1',
                    'CMCC-ESM2_r1i1p1f1_gn',
                    'MIROC6_r1i1p1f1_gn',
                    'NorESM2-LM_r1i1p1f1_gn',
                    'BCC-CSM2-MR_r1i1p1f1_gn',
                    'GFDL-CM4_r1i1p1f1_gr1',
                    'MPI-ESM1-2-LR_r1i1p1f1_gn',
                    'UKESM1-0-LL_r1i1p1f2_gn',
                    'ACCESS-CM2_r1i1p1f1_gn',
                    'INM-CM5-0_r1i1p1f1_gr1',
                    'MRI-ESM2-0_r1i1p1f1_gn',
                    'CanESM5-1_r1i1p1f1_gn',
                    'CNRM-CM6-1_r1i1p1f2_gr',
                    'EC-Earth3_r1i1p1f1_gr',
                    'NorESM2-MM_r1i1p1f1_gn']

        modelos_split = [m.split("_") for m in models]
        puntos_cuenca = pd.read_csv(path_project+'/01_CLIMA/Puntos_Cuenca.csv', index_col=0)

        transformer = Transformer.from_crs(25830, 4326, always_xy=True)  # EPSG:25830 to EPSG:4326
        longitudes, latitudes = transformer.transform(puntos_cuenca['COORDX'].values, puntos_cuenca['COORDY'].values)

        dictionary_Hist = {v: {model: {} for model in models} for v in ['pr', 'tasmax', 'tasmin']}
        dictionary_CC = {v: {rcp: {model: {} for model in models} for rcp in ['ssp245', 'ssp585']} for v in ['pr', 'tasmax', 'tasmin']}

        for sc in ['historical', 'ssp245', 'ssp585']:
            scenario = {'historical': 'historical', 'ssp245': 'ssp2_4_5', 'ssp585': 'ssp5_8_5'}[sc]
            for var in (['pr', 'tasmax', 'tasmin']):
                for m in modelos_split:
                    # nc_files = glob.glob(f"{path_data}/CMIP6/{var}/{scenario}/{var}_day_{m[0]}_{sc}_{m[1]}_{m[2]}*")
                    nc_files = glob.glob(f"{path_data}/CMIP6/{var}/{scenario}/{var}_day_{m[0]}_{sc}_{m[1]}_{m[2]}*")
                    if os.path.exists(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/{var}/{var}_month_{m[0]}_{sc}_{m[1]}_{m[2]}.csv"):
                        df = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/{var}/{var}_month_{m[0]}_{sc}_{m[1]}_{m[2]}.csv", index_col=0, parse_dates=True)
                    else:
                        os.makedirs(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/{var}/", exist_ok=True)
                        
                        time_index = pd.date_range(start='1950-01-01', end='2014-12-31', freq='M') if sc == 'historical' else pd.date_range(start='2015-01-01', end='2100-12-31', freq='M')
                        df = pd.DataFrame(index=time_index, columns=puntos_cuenca.index)
                        df.index = df.index.date - pd.offsets.MonthBegin(1)
                        
                        for nc_file in nc_files:
                            ds_day = xr.open_dataset(nc_file,decode_times=True)
                            ds_day = convertir_a_datetime64(ds_day)
                            if sc == 'historical':
                                fecha_inicio = '1995-01-01'
                                fecha_fin = '2014-12-31'
                            else:
                                fecha_inicio = '2015-01-01'
                                fecha_fin = '2100-12-31'
                            ds_day = ds_day.sel(time=slice(fecha_inicio, fecha_fin))
                            if ds_day['time'].size == 0:
                                # print(f"⚠️ Archivo sin datos entre las fechas establecidas: {nc_file}")
                                continue  # Salta al siguiente archivo
                            if var=='pr':
                                ds = ds_day[var].resample(time='M').reduce(np.nansum)*86400
                            else:
                                ds = ds_day[var].resample(time='M').mean()-273

                            time_vals = ds['time'].values
                            time_vals = pd.to_datetime(time_vals).normalize()- pd.offsets.MonthBegin(1)
                            
                            for idx, (lon, lat) in enumerate(zip(longitudes, latitudes)):
                                try:
                                    serie = ds.sel(lon=lon, lat=lat, method='nearest')
                                    df.loc[time_vals, puntos_cuenca.index[idx]] = serie.values
                                except Exception as e:
                                    print(f"Error en punto {idx} para archivo {nc_file}: {e}")
                            ds.close()
                            ds_day.close()
                        df.to_csv(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/{var}/{var}_month_{m[0]}_{sc}_{m[1]}_{m[2]}.csv")

                    if sc == 'historical':
                        dictionary_Hist[var][f'{m[0]}_{m[1]}_{m[2]}'] = df
                    else:
                        dictionary_CC[var][sc][f'{m[0]}_{m[1]}_{m[2]}'] = df

        with open(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/dict_hist.pickle", "wb") as f:
            pickle.dump(dictionary_Hist, f)
        with open(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/CMIP6/dict_CC.pickle", "wb") as f:
            pickle.dump(dictionary_CC, f)

        return models


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
    def __init__ (self,path_project,name_project,path_data,nombre_embalse,climate_change='CMIP6',logging =True):
        self.path_project    = path_project
        self.path_data       = path_data
        self.nombre_embalse  = nombre_embalse
        self.climate_change  = climate_change # CMIP6 or CORDEX
        if logging == False:
            tqdm.tqdm(disable =True)
            
        self.file_yml = path_project+'/'+name_project+'.yml'
    
    def extrac_climate_change(self):
        if self.climate_change=='CORDEX':
            self.models = extrac_climate_change_AEMET(self.path_project,self.path_data)
        elif self.climate_change=='CMIP6':
            self.models = extract_climate_change_CMIP6(self.path_project,self.path_data)
        else:
            raise ValueError(f"❌ Fuente desconocida: '{self.climate_change}'. Debe ser 'AEMET' o 'CMIP6'")
        
    def correccion_sesgo(self):
        """
        Con esta función se ecorrige el sesgo de las series de cambio climático.
        
        """
        puntos_cuenca = pd.read_csv(self.path_project+'/01_CLIMA/Puntos_Cuenca.csv', index_col=0)
        modelos_split = [m.split("_") for m in self.models]
        for var in ['pr','tasmax','tasmin']:
            if os.path.exists(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/'+var+'/')==False:
                os.makedirs(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/'+var+'/',exist_ok=True)
            if var == 'pr':
                variable = 'Precipitacion'
            elif var=='tasmax':
                variable = 'Temperatura_Maxima'
            elif var=='tasmin':
                variable = 'Temperatura_Minima'
            for m in modelos_split: 
                print('Corrigiendo serie histórica de la variable '+var+' del modelo '+m[0])
                Serie_hist = pd.read_csv(self.path_project+'/01_CLIMA/'+variable+'.csv',index_col=0,parse_dates=True)
                Serie_hist = Serie_hist.interpolate(method='linear', axis=1).ffill().bfill()
                Serie_hist.index = Serie_hist.index.date - pd.offsets.MonthBegin(1)
                if self.climate_change=='CORDEX':
                    name_file_hist = f'{var}_month_CORDEX_historical_{m[0]}_r1i1p1.csv'
                else:
                    name_file_hist = f'{var}_month_{m[0]}_historical_{m[1]}_{m[2]}.csv'

                serie_raw  = pd.read_csv(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}/{var}/{name_file_hist}',index_col=0,parse_dates=True)
                if var == 'pr':
                    serie_raw[serie_raw<=0.1] = 0.11
                if self.climate_change=='CORDEX':
                    Serie_hist_correc     = pd.DataFrame(index=pd.date_range(start='1976-01-01', end='2005-12-31', freq='M'),columns = puntos_cuenca.index)
                else:
                    Serie_hist_correc     = pd.DataFrame(index=pd.date_range(start='1995-01-01', end='2014-12-31', freq='M'),columns = puntos_cuenca.index)
                Serie_hist_correc.index = Serie_hist_correc.index.date - pd.offsets.MonthBegin(1)
                for c in puntos_cuenca.index:
                    if self.climate_change=='CORDEX':
                        serie_hist_time = serie_raw.loc['1976':'2005',str(c)].dropna()
                    else:
                        serie_hist_time = serie_raw.loc['1995':'2014',str(c)].dropna()

                
                    for month in range(1,13):

                        Serie_concat = pd.concat((Serie_hist.loc[Serie_hist.index.month==month,str(c)],serie_raw.loc[serie_raw.index.month==month,str(c)]),axis=1).dropna()

                        Bias_correction   =  bias_correction(Serie_concat.iloc[:,0].values.flatten().astype(float), 
                                                                            Serie_concat.iloc[:,1].values.flatten().astype(float), 
                                                                            serie_hist_time.loc[serie_hist_time.index.month==month].values.flatten().astype(float))

                        index_time = serie_hist_time.loc[serie_hist_time.index.month==month].index

                        if var == 'pr':
                                variable_2 = 'precipitation'
                        else:
                            variable_2 = 'temperature'

                        Serie_hist_correc.loc[index_time,c]  = Bias_correction.scaled_distribution_mapping(variable_2)
                        
                Serie_hist_correc.to_csv(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/{var}/{name_file_hist}')

                if self.climate_change=='CORDEX':
                    scenarios = ['rcp45','rcp85']
                else:
                    scenarios = ['ssp245','ssp585']   

                for rcp in scenarios:
                    if self.climate_change=='CORDEX':
                        name_file_fut = f'{var}_month_CORDEX_{rcp}_{m[0]}_r1i1p1.csv'
                    else:
                        name_file_fut = f'{var}_month_{m[0]}_{rcp}_{m[1]}_{m[2]}.csv'

                    print('Corrigiendo serie del escenario '+rcp+' de la variable '+var+' para el modelo '+m[0])
                    
                    if os.path.exists(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/{var}/{name_file_fut}'):
                        continue
                    else:
                    
                        Serie_CC = pd.read_csv(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}/{var}/{name_file_fut}',index_col=0,parse_dates=True)
                        
                        if var == 'pr':
                            Serie_CC[Serie_CC<=0.1] = 0.11

                        if self.climate_change=='CORDEX':
                            y_period = ['2006','2100']
                        else:
                            y_period = ['2015','2100']  

                        if self.climate_change=='CORDEX':
                            period = ['2011_2040','2041_2070','2071_2100']
                        else:
                            period = ['2021_2040','2041_2060','2061_2080','2081_2100']    
                        
                        Serie_CC_correc     = pd.DataFrame(index=pd.date_range(start=f'{y_period[0]}-01-01', end=f'{y_period[1]}-12-31', freq='M'),columns = puntos_cuenca.index)
                        Serie_CC_correc.index = Serie_CC_correc.index.date - pd.offsets.MonthBegin(1)

                        for c in puntos_cuenca.index:
                            
                            for month in range(1,13):

                                Serie_concat = pd.concat((Serie_hist.loc[Serie_hist.index.month==month,str(c)],serie_raw.loc[serie_raw.index.month==month,str(c)]),axis=1).dropna()

                                for p in period:
                                    year_ini = p.split('_')[0]
                                    year_fin = p.split('_')[1]

                                    Serie_CC_time = Serie_CC.loc[year_ini:year_fin,str(c)].dropna()

                                    Bias_correction   =  bias_correction(Serie_concat.iloc[:,0].values.flatten().astype(float), 
                                                                    Serie_concat.iloc[:,1].values.flatten().astype(float), 
                                                                    Serie_CC_time.loc[Serie_CC_time.index.month==month].values.flatten().astype(float))
                                    
                                    index_time = Serie_CC_time.loc[Serie_CC_time.index.month==month].index

                                    if var == 'pr':
                                        variable_2 = 'precipitation'
                                    else:
                                        variable_2 = 'temperature'

                                    Serie_CC_correc.loc[index_time,c]  = Bias_correction.scaled_distribution_mapping(variable_2)

                        Serie_CC_correc.to_csv(f'{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/{var}/{name_file_fut}')
                        
    def ejec_aport_cc(self):
        """
        Con esta función se simulan las aportaciones correspondientes a cada modelo y escenario de cambio climático.
        
        """
        
        file_model_Reg = self.path_project+'/04_REGRESION/'+self.nombre_embalse+'_'+'ANN'+'.sav'
        file_model_PCA = self.path_project+'/04_REGRESION/'+self.nombre_embalse+'_'+'PCA'+'.sav'
        file_yScaler = self.path_project+'/04_REGRESION/'+self.nombre_embalse+'_'+'yScaler'+'.sav'

        def create_name(var,rcp,m):
            if self.climate_change=='CORDEX':
                name_file_fut = f'{var}_month_CORDEX_{rcp}_{m[0]}_r1i1p1'
            else:
                name_file_fut = f'{var}_month_{m[0]}_{rcp}_{m[1]}_{m[2]}'
            return name_file_fut
        
        modelos_split = [m.split("_") for m in self.models]

        if self.climate_change=='CORDEX':
            esc = ['rcp45','rcp85']
        elif self.climate_change=='CMIP6':
            esc = ['ssp245','ssp585']
        
        for nmodel in tqdm.tqdm(modelos_split):
            print('### Ejecutando simulación de aportaciones modelo '+nmodel[0]+' ###')

            PREC_HIST = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/pr/{create_name('pr','historical',nmodel)}.csv",index_col=0,parse_dates=True)
            TMAX_HIST = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmax/{create_name('tasmax','historical',nmodel)}.csv",index_col=0,parse_dates=True)
            TMIN_HIST = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmin/{create_name('tasmin','historical',nmodel)}.csv",index_col=0,parse_dates=True)
            
            
            PREC_45 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/pr/{create_name('pr',esc[0],nmodel)}.csv",index_col=0,parse_dates=True)
            TMAX_45 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmax/{create_name('tasmax',esc[0],nmodel)}.csv",index_col=0,parse_dates=True)
            TMIN_45 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmin/{create_name('tasmin',esc[0],nmodel)}.csv",index_col=0,parse_dates=True)

            PREC_85 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/pr/{create_name('pr',esc[1],nmodel)}.csv",index_col=0,parse_dates=True)
            TMAX_85 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmax/{create_name('tasmax',esc[1],nmodel)}.csv",index_col=0,parse_dates=True)
            TMIN_85 = pd.read_csv(f"{self.path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{self.climate_change}_BIAS_CORRECTED/tasmin/{create_name('tasmin',esc[1],nmodel)}.csv",index_col=0,parse_dates=True)

            
            SIM_HIST = SIM_REGRESION(file_model_Reg,file_model_PCA,file_yScaler,
                                          PREC_HIST,
                                          TMAX_HIST,
                                          TMIN_HIST) 
            SIM_HIST.simulation()
            SIM_HIST.save_series(f"{create_name('Aportaciones','historical',nmodel)}",self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')
            
            
            SIM_45 = SIM_REGRESION(file_model_Reg,file_model_PCA,file_yScaler,
                                           PREC_45,
                                           TMAX_45,
                                           TMIN_45) 
            SIM_45.simulation()
            SIM_45.save_series(f"{create_name('Aportaciones',esc[0],nmodel)}",self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')

            SIM_85 = SIM_REGRESION(file_model_Reg,file_model_PCA,file_yScaler,
                                           PREC_85,
                                           TMAX_85,
                                           TMIN_85) 
            SIM_85.simulation()
            SIM_85.save_series(f"{create_name('Aportaciones',esc[1],nmodel)}",self.path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/')
            
            with open(self.file_yml) as file:
                params = yaml.load(file, Loader=yaml.FullLoader)

            params['ejec_aport_cc']   = 'Si'

            with open(self.file_yml, 'w') as file:
                documents = yaml.dump(params, file)
