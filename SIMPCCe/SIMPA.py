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
import xarray as xr

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

    def extract_flow_simpa(self, NombreEmbalse, Coordenadas, path_output):
        """
        Con esta función se pueden extraer los datos de aportaciones de SIMPA en un punto en concreto.
        Datos de Entrada:
        -----------------
        NombreEmbalse: str. Nombre del embalse en la forma que aparece en el índice
                       de Coordenadas
        Coordenadas: Pandas DataFrame. DataFrame con las coordenadas del punto de
                     vertido del embalse y de los puntos de vertido de las cuencas
                     que hay que sustraer.
        path_output: string. Directorio donde se quieren guardar las series temporales extraideas 

        Resultados:
        -----------------
        Flow: csv. Fichero csv con los resultados extraidos.

        """
        time=pd.date_range(start='1940-10-01',end='2018-09-30',freq='M')
        reference_time = pd.Timestamp("1940-10-01")

        Flow = pd.DataFrame(index = time, columns=Coordenadas.index)
        ds = xr.open_dataset(self.path_simpa+'/Aportaciones/Aportaciones_SIMPA_CEDEX.nc')
        for s, ss in enumerate(Coordenadas.index):
            Flow.loc[:,ss] = ds.aportacion.sel(y=Coordenadas.iloc[s].loc['COORDY'], x=Coordenadas.iloc[s].loc['COORDX'], method='nearest')


        Flow = pd.DataFrame({NombreEmbalse: Flow[NombreEmbalse] - Flow.drop(columns=NombreEmbalse).sum(axis=1)})

        Flow.to_csv(path_output+'/Aportaciones.csv')
