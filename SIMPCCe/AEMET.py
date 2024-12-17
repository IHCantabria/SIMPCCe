'''
La librería contiene las clases y funciones que permiten extraer los datos climáticos 
y extraer la cuenca de estudio aportante a un punto dado.
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''

import pandas as pd
#from osgeo import gdal
#from osgeo.gdalnumeric import *
#from osgeo.gdalconst import *
import os
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import xarray as xr
import tqdm
#from osgeo import gdal, ogr, osr
from math import floor
import cartopy.crs as ccrs
import scipy as sp
import warnings
import geopandas as gpd
from pyproj import Proj, transform
import cartopy.feature
warnings.filterwarnings('ignore')
from shapely import geometry, ops
import fiona
import xarray as xr
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Polygon, Circle

from pysheds.grid import Grid

def draw_screen_poly( lats, lons, ax):
    x, y = lons, lats
    xy = zip(x,y)
    poly = Polygon( list(xy),linestyle='-',closed=True,ec='red',lw=1,fill=False)
    ax.add_patch(poly)

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


class AEMET(object):
    """ 
    Esta función permite trabajar con datos de SPAIN 02 (AEMET)
    
    Datos de Entrada:
    ----------------
    path_project: string. Directorio del proyecto
    path_aemet:   string.Directorio donde se guardan los datos descargados.
        
    """
    def __init__ (self,path_aemet,path_project):
        self.path_aemet   = path_aemet
        self.path_project = path_project
        
    def extract_climate_aemet(self,X,Y):
        """
        Con esta función se pueden extraer los datos climáticos de SPAIN 02 (AEMET) en la cuenca aportante a un punto dado.
        Datos de Entrada:
        -----------------
        X          : float. Coordenada X en UTM ETRS89 Z30 del punto de desagüe
        Y          : float. Coordenada Y en UTM ETRS89 Z30 del punto de desagüe
        Resultados:
        -----------------
        Precipitation:       csv. Fichero csv con los resultados extraidos de precipitación.
        Temperatura_Maxima:  csv. Fichero csv con los resultados extraidos de temperatura máxima.
        Temperatura_Minima:  csv. Fichero csv con los resultados extraidos de temperatura mínima.
        Puntos_Cuenca:       csv. Coordenadas de los puntos distribuidos que definen la cuenca
        catchment:           shp. Shapefile que define la cuenca aportante
        """
        
        inProj = Proj(init='epsg:25830')
        outProj = Proj(init='epsg:4326')
        
        lon_C = transform(inProj,outProj,X,Y)[0]
        lat_C = transform(inProj,outProj,X,Y)[1]
        
        time=pd.date_range(start='1971-01-01',end='2015-12-31',freq='M')
        reference_time = pd.Timestamp("1971-01-01")
        
        # Extracción de la cuenca vertiente a un punto dado
        grid    = Grid.from_raster(self.path_project+'02_GIS/Str.tif')
        strr    = grid.read_raster(self.path_project+'02_GIS/Str.tif')
        
        x_2 = X
        y_2 = Y

        ncols = strr.shape[0]
        nrows = strr.shape[1]
        cellsize = strr.dy_dx[0]

        coordinates = np.fliplr(strr.coords)

        res = 15

        xx_vic = np.unique(coordinates.T[0])[::res]
        yy_vic = np.unique(coordinates.T[1])[::res]
        

        [XX_VIC, YY_VIC]=np.meshgrid(xx_vic,yy_vic)


        dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
        
        grid   = Grid.from_raster(self.path_project+'02_GIS/Fdr.tif') ## Change directory
        stream = Grid.from_raster(self.path_project+'02_GIS/Str.tif')      ## Change directory
        
        fdir = grid.read_raster(self.path_project+'02_GIS/Fdr.tif')
        strr  = stream.read_raster(self.path_project+'02_GIS/Str.tif')
        
        river=np.array(strr)
        river_rsp=np.reshape(river,(-1,1))
        pos_r=np.where(river_rsp==1)
        coordenadas=strr.coords.copy()
        coordenadas_river=np.fliplr(coordenadas[pos_r[0]])
        cellsize=np.abs(np.unique(np.sort(coordenadas.T[0]))[0]-np.unique(np.sort(coordenadas.T[0]))[1])
        coordenadas_river=coordenadas_river+[cellsize/2,-cellsize/2]
        dist=np.sqrt((x_2-coordenadas_river.T[0])**2+(y_2-coordenadas_river.T[1])**2)

        catch = grid.catchment(x=(coordenadas_river[np.argmin(dist)][0]-cellsize/2), y=(coordenadas_river[np.argmin(dist)][1]+cellsize/2),fdir=fdir, dirmap=dirmap, out_name='catch',
               recursionlimit=150000, xytype='label', nodata=-1)
        

        basin=np.flipud(np.array(catch).astype(float))
        basin=np.array(catch)
        
        

        B=np.ones((res,res))*1/(res*res)
        C_2=np.round(sp.signal.convolve2d(np.flipud(catch),B, fillvalue=-1,mode='valid'))
        C_2=C_2[0::res,0::res]

        pos_r=np.where(C_2>0)
        coordenadas_basin=np.stack((XX_VIC[pos_r[0],pos_r[1]],YY_VIC[pos_r[0],pos_r[1]])).T
        
        coordenadas_basin_DF = pd.DataFrame(coordenadas_basin,index=np.arange(1,len(coordenadas_basin)+1),columns = ['COORDX','COORDY'])
        coordenadas_basin_DF.to_csv(self.path_project+'01_CLIMA'+'/Puntos_Cuenca.csv')

        print('Coordenada X: '+str(coordenadas_river[np.argmin(dist)][0]), 'Coordenada Y: '+str(coordenadas_river[np.argmin(dist)][1]))

        lons = transform(inProj,outProj,coordenadas_basin.T[0],coordenadas_basin.T[1])[0]
        lats = transform(inProj,outProj,coordenadas_basin.T[0],coordenadas_basin.T[1])[1]
        
        grid.clip_to(catch)
        
        shapes = grid.polygonize()
        
        schema = {
            'geometry': 'Polygon',
            'properties': {'LABEL': 'float:16'}
        }

        with fiona.open(self.path_project+'02_GIS/catchment.shp', 'w',
                        driver='ESRI Shapefile',
                        crs=grid.crs.srs,
                        schema=schema) as c:
            i = 0
            for shape, value in shapes:
                rec = {}
                rec['geometry'] = shape
                rec['properties'] = {'LABEL' : str(value)}
                rec['id'] = str(i)
                c.write(rec)
                i += 1
        
        gdf = gpd.read_file(self.path_project+'02_GIS/catchment.shp')  
        gd2 = gdf.to_crs(epsg=4326)
        
        vector_bound_coordinates= gd2['geometry']
        Extent = [vector_bound_coordinates.bounds.minx.min(),
          vector_bound_coordinates.bounds.maxx.max(),
          vector_bound_coordinates.bounds.miny.min(),
          vector_bound_coordinates.bounds.maxy.max()]
        
        
        #[image, xmin, xmax, ymin, ymax] = google_maps(lon_C=np.mean([np.min(lons),np.max(lons)]), lat_C =np.mean([np.min(lats),np.max(lats)]) ,size=[640, 640], zoom=10, scale=1, api_key = 'AIzaSyDAYDmAVdeZ3LOthK1_25qRWr7L25VomQY')
        
        import cartopy.feature

        fig, ax = plt.subplots(figsize=(7,5),subplot_kw=dict(projection=ccrs.PlateCarree()))


        # fname=path_GIS+'/catchment.shp'

        # shape_feature = ShapelyFeature(Reader(fname).geometries(),
        #                                 ccrs.PlateCarree(),facecolor='grey',edgecolor='black')


        #ax.add_feature(shape_feature)
        #ax.add_feature(rivers, linewidth=1)
        ax.plot(lons,lats,'.b','Puntos cuenca')
        ax.plot(lon_C,lat_C,'.r',markersize=12,label = 'Punto Desagüe')
        ax.set_extent(np.array(Extent))
        #co=ax.pcolor(XX, YY, z_OK_grid, cmap='Blues')
        #co2=ax.contour(XX, YY, z_OK_grid, 12, colors='darkblue',alpha=0.6)
        gl = ax.gridlines(draw_labels=True, alpha=0.2)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'gray'}
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'size': 20, 'color': 'gray'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        gd2.plot(ax=ax,facecolor="grey",edgecolor='black')
        fig.savefig(self.path_project+'/07_INFORME/Figuras/'+'Cuenca_Estudio.png',bbox_inches='tight')
        fig.tight_layout()
        #plt.clabel(co2, inline=True, fontsize=12)
        #fig.colorbar(co,label='mm/año')
        #draw_circle(Centroides.POINT_X[c].astype(float),Centroides.POINT_Y[c].astype(float),1.2)
        # ax.scatter(Estaciones_Tarija_Prec[2].values,Estaciones_Tarija_Prec[3].values, c=Estaciones_Tarija_Prec[6], s=30, cmap='Blues', alpha=0.5)
        #plt.title('Precipitacion media anual (mm)')
        #plt.savefig(path_documents+'Precipitacion_distribuida_OK.png',bbox_inches='tight',dpi=350)
        #plt.savefig(path_documents+'Localizacion_zoom_'+'Centroide_'+str(c)+'.png'+'.png',bbox_inches='tight',dpi=350)

        #fig, ax= plt.subplots(figsize=(7, 5),subplot_kw=dict(projection=ccrs.PlateCarree())) 
        #ax.coastlines()
        #print(lon_C,lat_C)
        #ax.plot(lons,lats,'.b','Puntos cuenca')
        #ax.plot(lon_C,lat_C,'.r',markersize=12,label = 'Punto Desagüe')
        #gd2.plot(ax=ax,facecolor="none",edgecolor='black', lw=2)
        
        #ax.imshow(image,extent=[xmin, xmax, ymin, ymax],origin = 'upper',alpha = 0.7)
        
        # Se extraen los datos climáticos en los puntos de la cuenca obtenida anteriormente
        Precipitation       = pd.DataFrame(index = time, columns=np.arange(1, len(coordenadas_basin)+1))
        Temperatura_Maxima  = pd.DataFrame(index = time, columns=np.arange(1, len(coordenadas_basin)+1))
        Temperatura_Minima  = pd.DataFrame(index = time, columns=np.arange(1, len(coordenadas_basin)+1))
        
        
        prec_day_nc   = xr.open_dataset(self.path_aemet+'/Spain02_v5.0_DD_010reg_aa3d_pr.nc')
        tasmax_day_nc = xr.open_dataset(self.path_aemet+'/Spain02_v5.0_DD_010reg_aa3d_tasmax.nc')
        tasmin_day_nc = xr.open_dataset(self.path_aemet+'/Spain02_v5.0_DD_010reg_aa3d_tasmin.nc')

        prec_nc   = prec_day_nc.resample(time='M').sum(min_count=1)
        tasmax_nc = tasmax_day_nc.resample(time='M').mean()
        tasmin_nc = tasmin_day_nc.resample(time='M').mean()


        for s in range(len(coordenadas_basin)):
            mx = lons[s]
            my = lats[s]
            
            prec   = prec_nc.sel(lon=mx,lat=my,method="nearest")
            tasmax = tasmax_nc.sel(lon=mx,lat=my,method="nearest")
            tasmin = tasmin_nc.sel(lon=mx,lat=my,method="nearest")

            Precipitation.iloc[:,s]      = prec.pr.data
            Temperatura_Maxima.iloc[:,s] = tasmax.tasmax.data
            Temperatura_Minima.iloc[:,s] = tasmin.tasmin.data

        Precipitation.to_csv(self.path_project+'01_CLIMA/Precipitacion.csv')
        Temperatura_Maxima.to_csv(self.path_project+'01_CLIMA/Temperatura_Maxima.csv')
        Temperatura_Minima.to_csv(self.path_project+'01_CLIMA/Temperatura_Minima.csv')
        
        fig.tight_layout()
        
        
        from matplotlib.patches import Polygon, Circle
        fig2, ax2= plt.subplots(figsize=(15, 10),subplot_kw=dict(projection=ccrs.PlateCarree())) 

        inProj = Proj(init='epsg:25830')
        outProj = Proj(init='epsg:4326')
        
        lon_C = transform(inProj,outProj,X,Y)[0]
        lat_C = transform(inProj,outProj,X,Y)[1]
        
        frios=self.path_project+'02_GIS/RIOS_WGS84.shp' #sys._MEIPASS
        shape_feature_rios = ShapelyFeature(Reader(frios).geometries(),
                                        ccrs.PlateCarree())
        ax2.add_feature(shape_feature_rios, facecolor='none',edgecolor='b',alpha=0.5)

        fbasin=self.path_project+'02_GIS/DemarcacionesHidrograficas_WGS84.shp' #sys._MEIPASS
        shape_feature_basin = ShapelyFeature(Reader(fbasin).geometries(),
                                        ccrs.PlateCarree())
        ax2.add_feature(shape_feature_basin, facecolor='none',edgecolor='k')


        ax2.set_extent([-9.7,4.7,35.5,43.9])
        ax2.add_feature(cfeature.LAND,facecolor="grey",alpha=0.6)
        ax2.add_feature(cfeature.COASTLINE,edgecolor='black')
        lats = [ lat_C-1,lat_C+1,lat_C+1, lat_C-1 ]
        lons = [lon_C-1, lon_C-1, lon_C+1,lon_C+1]
        im0 = ax2.plot(lon_C,lat_C,'.r',markersize=20)
        draw_screen_poly( lats, lons, ax2)
        gl = ax2.gridlines(draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 20, 'color': 'gray'}
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'size': 20, 'color': 'gray'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
        plt.close(fig2)
        fig2.savefig(self.path_project+'/07_INFORME/Figuras/'+'Localizacion_Estudio.png',bbox_inches='tight')

        