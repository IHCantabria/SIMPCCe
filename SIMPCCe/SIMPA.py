'''
La librería contiene las clases y funciones que permiten extraer los datos de aportaciones de SIMPA 
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''

import pandas as pd
from osgeo import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import os
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import xarray as xr
import tqdm
from osgeo import gdal, ogr, osr
from math import floor
import cartopy.crs as ccrs
import scipy as sp
import warnings
import geopandas as gpd
from pyproj import Proj, transform
warnings.filterwarnings('ignore')
from shapely import geometry, ops
import fiona

from pysheds.grid import Grid

def read_asc(file):
    name = list()
    data = list()
    c = 0
    with open(file) as input_file:
        for line in input_file:
            c = c + 1
            if c < 7:
                a, b = (item.strip() for item in line.split(' ', 1))
                name.append(a)
                data.append(b) 
    ncols         =   int(int(data[0]))
    nrows         =   int(int(data[1]))
    xllcorner     =   np.float(data[2])
    yllcorner     =   np.float(data[3])
    cellsize      =   np.float(data[4])
    return(ncols,nrows,xllcorner,yllcorner,cellsize)


class SIMPA(object):
    """ 
    Esta función permite trabajar con datos de SIMPA
    
    Datos de Entrada:
    ----------------
    path_simpa:   string. Directorio donde se encuentran los datos descargados de SIMPA.
        
    """
    def __init__ (self,path_simpa):
        self.path_simpa   = path_simpa

    def extract_flow_simpa(self,coord,path_output):
        """
        Con esta función se pueden extraer los datos de aportaciones de SIMPA en un punto en concreto.
        Datos de Entrada:
        -----------------
        coord:       dataframe. Tabla con las coordenadas de los puntos en los que se desea obtener las aportaciones
        path_output: string. Directorio donde se quieren guardar las series temporales extraideas 

        Resultados:
        -----------------
        Flow: csv. Fichero csv con los resultados extraidos.

        """
        time=pd.date_range(start='1940-10-01',end='2015-12-31',freq='M')
        reference_time = pd.Timestamp("1940-10-01")

        Flow = pd.DataFrame(index = time, columns=coord.index)
        for i,ii in enumerate(tqdm.tqdm(time)):
            #flow[:,:,i]=np.flipud(np.loadtxt(path+'acaes'+str(ii.year)+'_'+str(ii.month)+'.asc',skiprows=6))
            ds = gdal.Open(self.path_simpa+'/Aportaciones/acaesh'+str(ii.year)+'_'+str(ii.month)+'.asc', gdal.GA_ReadOnly)
            gt   = ds.GetGeoTransform()

            for s, ss in enumerate(coord.index):
                value_month = []
                mx = coord.iloc[s].loc['COORDX']
                my = coord.iloc[s].loc['COORDY']

                px = floor((mx - gt[0]) / gt[1]) #x pixel
                py = floor((my - gt[3]) / gt[5]) #y pixel

                intval=ds.ReadAsArray(px,py,1,1)
                Flow.loc[ii,ss] = intval[0][0]
            del ds

        Flow.to_csv(path_output+'/Aportaciones.csv')