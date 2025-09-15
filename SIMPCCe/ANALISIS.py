'''
La librería contiene las funciones que permiten realizar el análisis final de los resultados
junto con la contrucción de las plantillas del proyecto.
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''

from PIL import Image
from io import BytesIO
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon, Circle
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pyproj import Proj, transform
import geopandas as gpd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import tqdm
import scipy.stats as stats

models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']


def create_name(climate_change,var,rcp,m):
    if climate_change=='CORDEX':
        name_file_fut = f'{var}_month_CORDEX_{rcp}_{m[0]}_r1i1p1'
    else:
        name_file_fut = f'{var}_month_{m[0]}_{rcp}_{m[1]}_{m[2]}'
    return name_file_fut


def plot_climograma(path_project):
    
    prec = pd.read_csv(path_project+'/01_CLIMA/Precipitacion.csv',index_col=0, parse_dates=True)
    tmax = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Maxima.csv',index_col=0, parse_dates=True)
    tmin = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Minima.csv',index_col=0, parse_dates=True)

    prec_hist = prec.mean(axis=1)
    tmax_hist = tmax.mean(axis=1)
    tmin_hist = tmin.mean(axis=1)
    
    import matplotlib.ticker as mticker
    
    tmed_hist = ((tmax_hist+tmin_hist)/2)

    temp_group=tmed_hist.groupby(lambda x: x.month)
    temp_group_min=tmin_hist.groupby(lambda x: x.month)
    temp_group_max=tmax_hist.groupby(lambda x: x.month)

    Temperature_month=temp_group.mean()
    Temperature_month_min=temp_group_min.mean()
    Temperature_month_max=temp_group_max.mean()

    Temperature_month_min_abs=temp_group_min.min()
    Temperature_month_max_abs=temp_group_max.max()

    precp_group=prec_hist.resample('M').sum().groupby(lambda x: x.month)
    Precpitacion_month=precp_group.mean()

    fig, ax = plt.subplots(figsize=(16,8))
    ax.bar(range(1,13),Precpitacion_month, align='center', label = 'Precipitación')
    x=np.arange(1,13,1)
    ax2=ax.twinx()
    ax2.plot(range(1,13),Temperature_month,'r-', label = 'Temperatura')

    # ax2.fill_between(np.arange(1,13), Temperature_month_min_abs.values.T[0],Temperature_month_max_abs.values.T[0], facecolor='none',edgecolor='k', label='Temperatura mínima y máxima media')

    ax2.plot(np.arange(1,13),Temperature_month_min_abs,'k--')
    ax2.plot(np.arange(1,13),Temperature_month_max_abs,'k--',label='Temperatura mínima y máxima absoluta')
    #ax2.fill_between(np.arange(1,13), Temperature_month_min.values.T[0],Temperature_month_max.values.T[0], color='red', alpha=0.2, label='Temperatura mínima y máxima media')
    my_xticks = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    nticks=10
    ax.yaxis.set_major_locator(mticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(mticker.LinearLocator(nticks))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax2.set_ylim(np.min(Temperature_month_min_abs)-20,np.max(Temperature_month_max_abs)+5)
    ax.set_ylim(0,round(np.max(Precpitacion_month))+10)
    plt.xticks(x, my_xticks, fontsize=20)
    legnd = ['Precipitación']
    # legend_2=['Temperatura','Temperatura mínima y máxima absoluta','Temperatura mínima y máxima media']
    #ax.legend(legnd, loc='upper left', fontsize=25)
    #ax2.legend(loc=1,fontsize=25)
    
    lines = []
    labels = []
    
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    
    axLine, axLabel = ax2.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    

    fig.legend(lines, labels,           
               loc = 8,ncol=3,fontsize = 20)
    fig.tight_layout(pad=8)
    # ax2.legend(legend_2, loc='upper right', fontsize=25)
    ax.set_ylabel('Precipitación (mm)',fontsize=25)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)
    ax2.set_ylabel('Temperatura (ºC)',fontsize=25)
    fig.suptitle('Climograma', fontsize=25, y = 0.94)
    fig.savefig(path_project+'/07_INFORME/Figuras/Climograma'+'.png',bbox_inches='tight',dpi=350)
    
    
def plot_clima_cuenca(path_project):
    
    prec = pd.read_csv(path_project+'/01_CLIMA/Precipitacion.csv',index_col=0, parse_dates=True)
    tmax = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Maxima.csv',index_col=0, parse_dates=True)
    tmin = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Minima.csv',index_col=0, parse_dates=True)

    ####### Precipitación #####

    fig, ax = plt.subplots(figsize=(20,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
    Puntos_cuenca = pd.read_csv(path_project+'/01_CLIMA/Puntos_Cuenca.csv',index_col=0)

    inProj = Proj(init='epsg:25830')
    outProj = Proj(init='epsg:4326')

    Puntos_cuenca['Lon'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[0]).astype(float)
    Puntos_cuenca['Lat'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[1]).astype(float)

    gdf = gpd.read_file(path_project+'/02_GIS/catchment.shp')  
    gd2 = gdf.to_crs(epsg=4326)

    vector_bound_coordinates= gd2['geometry']
    Extent = vector_bound_coordinates.bounds
    Extent= Extent.values[0]
    Extent = [Extent[0],Extent[2],Extent[1],Extent[3]]
    
    
    cmap_pr = plt.cm.Blues  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap_pr(i) for i in range(cmap_pr.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap_pr = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap_pr.N)

    # define the bins and normalize
    bounds = np.arange(prec.resample('A').sum().mean().min()-20, prec.resample('A').sum().mean().max(), 15).round(1)
    norm_pr = matplotlib.colors.BoundaryNorm(bounds, cmap_pr.N)
    
    cmap_tmax = plt.cm.hot_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap_tmax(i) for i in range(cmap_tmax.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap_tmax = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap_tmax.N)

    # define the bins and normalize
    bounds = np.linspace(tmax.resample('A').mean().mean().min()-1.5, tmax.resample('A').mean().mean().max(), 15).round(1)
    norm_tmax = matplotlib.colors.BoundaryNorm(bounds, cmap_tmax.N)
    
    cmap_tmin = plt.cm.hot_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap_tmin(i) for i in range(cmap_tmin.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap_tmin = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap_tmin.N)

    # define the bins and normalize
    bounds =  np.linspace(tmin.resample('A').mean().mean().min()-1.5, tmin.resample('A').mean().mean().max(), 15).round(1)
    norm_tmin = matplotlib.colors.BoundaryNorm(bounds, cmap_tmin.N)
    
    
    #ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    #ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    im=ax.scatter(Puntos_cuenca.Lon,Puntos_cuenca.Lat,c=prec.resample('A').sum().mean().values,linewidth=8,cmap='Blues',norm= norm_pr, label='Datos de precipitaón diarios')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gd2.plot(ax=ax,facecolor='None',edgecolor='black')
    gl.xlabels_top = False
    gl.ylabels_left = False

    # gl.ylocator = mticker.FixedLocator([20,30,40,50,60,70,80])
    # gl.xlocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30, 40,45])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 0.5, 'color': 'gray'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 0.5, 'color': 'gray'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    #ax.set_extent([-4.9, -3.1, 42.7, 43.6], crs=ccrs.PlateCarree())

    #ax.text(-4.8, 43.4,u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation = 0)
    cb = fig.colorbar(im,orientation='horizontal',pad=0.06,shrink=0.47)
    cb.set_label(label='Precipitación anual (mm)', size=15)
    cb.ax.tick_params(labelsize=12)

    plt.savefig(path_project+'/07_INFORME/Figuras/Precipitacion_Basin'+'.png',bbox_inches='tight',dpi=350)

    ####### Temperatura Máxima #####

    fig, ax = plt.subplots(figsize=(20,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
    Puntos_cuenca = pd.read_csv(path_project+'/01_CLIMA/Puntos_Cuenca.csv',index_col=0)

    inProj = Proj(init='epsg:25830')
    outProj = Proj(init='epsg:4326')

    Puntos_cuenca['Lon'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[0]).astype(float)
    Puntos_cuenca['Lat'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[1]).astype(float)

    gdf = gpd.read_file(path_project+'/02_GIS/catchment.shp')  
    gd2 = gdf.to_crs(epsg=4326)

    vector_bound_coordinates= gd2['geometry']
    Extent = vector_bound_coordinates.bounds
    Extent= Extent.values[0]
    Extent = [Extent[0],Extent[2],Extent[1],Extent[3]]

    #ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    #ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    #ax.add_feature(shape_feature)
    im=ax.scatter(Puntos_cuenca.Lon,Puntos_cuenca.Lat,c=tmax.resample('A').mean().mean().values,linewidth=8,cmap='hot_r',norm=norm_tmax,label='Temperatura Máxima')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gd2.plot(ax=ax,facecolor='None',edgecolor='black')
    gl.xlabels_top = False
    gl.ylabels_left = False

    # gl.ylocator = mticker.FixedLocator([20,30,40,50,60,70,80])
    # gl.xlocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30, 40,45])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 0.5, 'color': 'gray'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 0.5, 'color': 'gray'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    #ax.set_extent([-4.9, -3.1, 42.7, 43.6], crs=ccrs.PlateCarree())

    #ax.text(-4.8, 43.4,u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation = 0)
    cb = fig.colorbar(im,orientation='horizontal',pad=0.06,shrink=0.47)
    cb.set_label(label='Temperatura máxima media anual (ºC)', size=15)
    cb.ax.tick_params(labelsize=12)
    plt.savefig(path_project+'/07_INFORME/Figuras/Temperatura_Max_Basin'+'.png',bbox_inches='tight',dpi=350)

    ####### Temperatura Mínima #####

    fig, ax = plt.subplots(figsize=(20,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
    Puntos_cuenca = pd.read_csv(path_project+'/01_CLIMA/Puntos_Cuenca.csv',index_col=0)

    inProj = Proj(init='epsg:25830')
    outProj = Proj(init='epsg:4326')

    Puntos_cuenca['Lon'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[0]).astype(float)
    Puntos_cuenca['Lat'] = np.array(transform(inProj,outProj,Puntos_cuenca.loc[:,'COORDX'],Puntos_cuenca.loc[:,'COORDY'])[1]).astype(float)

    gdf = gpd.read_file(path_project+'/02_GIS/catchment.shp')  
    gd2 = gdf.to_crs(epsg=4326)

    vector_bound_coordinates= gd2['geometry']
    Extent = vector_bound_coordinates.bounds
    Extent= Extent.values[0]
    Extent = [Extent[0],Extent[2],Extent[1],Extent[3]]

    #ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    #ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    #ax.add_feature(shape_feature)
    im=ax.scatter(Puntos_cuenca.Lon,Puntos_cuenca.Lat,c=tmin.resample('A').mean().mean().values,linewidth=8,cmap='hot_r',norm=norm_tmin,label='Temperatura Mínima')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gd2.plot(ax=ax,facecolor='None',edgecolor='black')
    gl.xlabels_top = False
    gl.ylabels_left = False

    # gl.ylocator = mticker.FixedLocator([20,30,40,50,60,70,80])
    # gl.xlocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30, 40,45])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 0.5, 'color': 'gray'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 0.5, 'color': 'gray'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}
    #ax.set_extent([-4.9, -3.1, 42.7, 43.6], crs=ccrs.PlateCarree())

    #ax.text(-4.8, 43.4,u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation = 0)
    cb = fig.colorbar(im,orientation='horizontal',pad=0.06,shrink=0.47)
    cb.set_label(label='Temperatura mínima media anual (ºC)', size=15)
    cb.ax.tick_params(labelsize=12)
    plt.savefig(path_project+'/07_INFORME/Figuras/Temperatura_Min_Basin'+'.png',bbox_inches='tight',dpi=350)
    
    
def plot_cambios_regimen_medio(path_project,models, climate_change):  
    #sns.set_style("white")
    #sns.set_context("poster")
    path_climate_change = f'{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}_BIAS_CORRECTED/'
    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        hist_period = ['1976','2005']
        periodos_N  = ['2011_2040','2041_2070','2071_2100']
        labels      = ['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
        labelsize   = 20
    elif climate_change=='CMIP6':
        esce       = ['ssp245','ssp585']
        hist_period = ['1995','2014']
        periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']
        labels      = ['SSP 245 2021-2040','SSP 245 2041-2060','SSP 245 2061-2080','SSP 245 2081-2100',
                       'SSP 585 2021-2040','SSP 585 2041-2060','SSP 585 2061-2080','SSP 585 2081-2100']
        labelsize = 16
    
    
    prec = pd.read_csv(path_project+'/01_CLIMA/Precipitacion.csv',index_col=0, parse_dates=True).mean(axis=1).loc[hist_period[0]:hist_period[1]]
    tmax = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Maxima.csv',index_col=0, parse_dates=True).mean(axis=1).loc[hist_period[0]:hist_period[1]]
    tmin = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Minima.csv',index_col=0, parse_dates=True).mean(axis=1).loc[hist_period[0]:hist_period[1]]
    
    fig, ax = plt.subplots(nrows=len(periodos_N), ncols=2 ,figsize=(15, 16))
    col=0
    row=0
    l=0
    for i,rcp in enumerate(esce):
        for j, p in enumerate(periodos_N):
            factor_prec = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_tmax = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_tmin = pd.DataFrame(index=np.arange(1,13),columns=models)
            
            for nmod in models:
                modelo_split = nmod.split("_")
                prec_c   =  pd.read_csv(f"{path_climate_change}/pr/{create_name(climate_change,'pr',rcp,modelo_split)}.csv",index_col=0, parse_dates=True).mean(axis=1)
                tasmax_c =  pd.read_csv(f"{path_climate_change}/tasmax/{create_name(climate_change,'tasmax',rcp,modelo_split)}.csv",index_col=0, parse_dates=True).mean(axis=1)
                tasmin_c =  pd.read_csv(f"{path_climate_change}/tasmin/{create_name(climate_change,'tasmin',rcp,modelo_split)}.csv",index_col=0, parse_dates=True).mean(axis=1)
                
                prec_c   = prec_c.loc[p.split('_')[0] : p.split('_')[1]]
                tasmax_c = tasmax_c.loc[p.split('_')[0] : p.split('_')[1]]
                tasmin_c = tasmin_c .loc[p.split('_')[0]: p.split('_')[1]]
                
                for m in range(1,13):
                    prec_m   = prec_c[prec_c.index.month==m]
                    tasmax_m = tasmax_c[tasmax_c.index.month==m]
                    tasmin_m = tasmin_c[tasmin_c.index.month==m]

                    factor_prec.loc[m,nmod] = prec_m.mean()*100/prec[prec.index.month==m].mean()-100
                    factor_tmax.loc[m,nmod] = tasmax_m.mean() - tmax[tmax.index.month==m].mean()
                    factor_tmin.loc[m,nmod] = tasmin_m.mean() - tmin[tmin.index.month==m].mean()
            #print(rcp+'  '+str(p[0])+'-'+str(p[1]))
            #print('Factor medio prec: '+str(np.mean(factor_prec)))
            #print('Factor medio tmax: '+str(np.mean(factor_tmax)))
            #print('Factor medio tmin: '+str(np.mean(factor_tmin)))
            l1=ax[j,col].bar(np.arange(1,13),factor_prec.mean(axis=1).values,color = 'dodgerblue',label='Precipitación')[0]
            ax2= ax[j,col].twinx() 
            l2=ax2.plot(np.arange(1,13),factor_tmax.mean(axis=1).values,linestyle='-', marker='o',color='red', label='Temperatura máxima')[0]
            l3=ax2.plot(np.arange(1,13),factor_tmin.mean(axis=1).values,linestyle='-',marker='o', color='darkblue',label='Temperatura mínima')[0]

            ax[j,col].set_ylim(-100,100)
            ax[j,col].tick_params(axis = 'both', which = 'major', labelsize = labelsize)
            ax2.set_ylim(-10,10)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = labelsize)

            ax[j,col].set_ylabel('Cambios en precipitación (%)',fontsize = labelsize)
            ax2.set_ylabel('Cambios en temperatura (ºC)',fontsize = labelsize)

            ax[j,col].set_title(labels[l],fontsize = 22)
            ax[j,col].set_xticks(np.arange(1,13))
            ax[j,col].grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
            l=l+1
        col=col+1


    # Recoger elementos únicos de leyenda usando fig.axes
    unique_legend = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in unique_legend:
                unique_legend[l] = h

    # Añadir leyenda global sin duplicados
    fig.legend(
        handles=list(unique_legend.values()),
        labels=list(unique_legend.keys()),
        loc=8,
        ncol=3,
        fontsize=labelsize)

    fig.tight_layout(pad=5)
    fig.suptitle('Cambios en el régimen medio mensual', fontsize=25,y=0.99)
    fig.savefig(path_project+'/07_INFORME/Figuras/Cambios_Clima.png',bbox_inches='tight',dpi=350)
    
    
def serie_climate_change(path_project,var,models,climate_change):
    def safe_fill_between(ax, x, y1, y2, **kwargs):
        x = pd.to_datetime(x)
        y1 = pd.to_numeric(y1, errors='coerce')
        y2 = pd.to_numeric(y2, errors='coerce')
        mask = ~(y1.isna() | y2.isna())
        ax.fill_between(x[mask], y1[mask], y2[mask], **kwargs)

    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1950','2014']
        fut_period  = ['2015','2100']
   
    pickle_in_hist = open(f'{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}/dict_hist.pickle',"rb")
    dict_hist = pickle.load(pickle_in_hist)

    pickle_in_CC = open(f'{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}/dict_CC.pickle',"rb")
    dict_CC = pickle.load(pickle_in_CC)

    dataframe_hist=pd.DataFrame(index=pd.date_range(start=f'{hist_period[0]}-01-01',end=f'{hist_period[1]}-12-31',freq='M'),columns=models)
    dataframe_hist.index = dataframe_hist.index.date - pd.offsets.MonthBegin(1)

    dataframe_45=pd.DataFrame(index=pd.date_range(start=f'{fut_period[0]}-01-01',end=f'{fut_period[1]}-12-31',freq='M'),columns=models)
    dataframe_45.index = dataframe_45.index.date - pd.offsets.MonthBegin(1)

    dataframe_85=pd.DataFrame(index=pd.date_range(start=f'{fut_period[0]}-01-01',end=f'{fut_period[1]}-12-31',freq='M'),columns=models)
    dataframe_85.index = dataframe_85.index.date - pd.offsets.MonthBegin(1)
    for i in tqdm.tqdm(models):
        d0=dict_hist[var][i]
        d1=dict_CC[var][esce[0]][i]
        d2=dict_CC[var][esce[0]][i]
        d3=dict_CC[var][esce[0]][i]

        d4=dict_CC[var][esce[1]][i]
        d5=dict_CC[var][esce[1]][i]
        d6=dict_CC[var][esce[1]][i]

        d00=d0.mean(axis=1)
        d10=d1.mean(axis=1)
        d20=d2.mean(axis=1)
        d30=d3.mean(axis=1)
        d40=d4.mean(axis=1)
        d50=d5.mean(axis=1)
        d60=d6.mean(axis=1)

        d00.index = pd.to_datetime(d00.index.astype(str),errors='coerce'); d00 = d00[~d00.index.isnull()];
        d10.index = pd.to_datetime(d10.index.astype(str),errors='coerce'); d10 = d10[~d10.index.isnull()];
        d20.index = pd.to_datetime(d20.index.astype(str),errors='coerce'); d20 = d20[~d20.index.isnull()];
        d30.index = pd.to_datetime(d30.index.astype(str),errors='coerce'); d30 = d30[~d30.index.isnull()];
        d40.index = pd.to_datetime(d40.index.astype(str),errors='coerce'); d40 = d40[~d40.index.isnull()];
        d50.index = pd.to_datetime(d50.index.astype(str),errors='coerce'); d50 = d50[~d50.index.isnull()];
        d60.index = pd.to_datetime(d60.index.astype(str),errors='coerce'); d60 = d60[~d60.index.isnull()];

        dataframe_hist.loc[d00.index,i]=d00.values
        dataframe_45.loc[d10.index,i]=d10.values
        dataframe_45.loc[d20.index,i]=d20.values
        dataframe_45.loc[d30.index,i]=d30.values

        dataframe_85.loc[d40.index,i]=d40.values
        dataframe_85.loc[d50.index,i]=d50.values
        dataframe_85.loc[d60.index,i]=d60.values

    dataframe_hist.index=pd.to_datetime(dataframe_hist.index)
    dataframe_45.index=pd.to_datetime(dataframe_45.index)
    dataframe_85.index=pd.to_datetime(dataframe_85.index)
    
    if var=='Prec':
        dataframe_hist_year=dataframe_hist.dropna().resample('A').sum()
        dataframe_45_year=dataframe_45.dropna().resample('A').sum()
        dataframe_85_year=dataframe_85.dropna().resample('A').sum()
    else:
        dataframe_hist_year=dataframe_hist.dropna().resample('A').mean()
        dataframe_45_year=dataframe_45.dropna().resample('A').mean()
        dataframe_85_year=dataframe_85.dropna().resample('A').mean()
        
    dataframe_hist_year.index   = dataframe_hist_year.index + pd.offsets.YearBegin(1)  
    dataframe_45_year.index = dataframe_45_year.index - pd.offsets.YearBegin(1)
    dataframe_85_year.index = dataframe_85_year.index - pd.offsets.YearBegin(1)
        
        
    value_max = np.max([dataframe_hist_year.dropna().max(),dataframe_45_year.dropna().max(),dataframe_85_year.dropna().max()])
    value_min = np.min([dataframe_hist_year.dropna().min(),dataframe_45_year.dropna().min(),dataframe_85_year.dropna().min()])
    
    if var=='Prec':
        vmin = value_min-10
        vmax = value_max+10
    else:
        vmin = value_min-1.5
        vmax = value_max+1.5
        
    fig, ax = plt.subplots(figsize=(10, 5))
    dataframe_hist_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='darkgray',ax=ax,label='Hist')
    dataframe_45_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='blue',ax=ax,label=label_esce[0])
    dataframe_85_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='red',ax=ax,label=label_esce[1])
    if climate_change=='CORDEX':
        ax.vlines('2006-01-01', vmin, vmax, 'k', linestyle = '-', linewidth = 2)
    elif climate_change=='CMIP6':
        ax.vlines('2015-01-01', vmin, vmax, 'k', linestyle = '-', linewidth = 2)
    safe_fill_between(ax,dataframe_hist_year.index, dataframe_hist_year.astype(float).quantile(0.25,axis=1),dataframe_hist_year.quantile(0.75,axis=1), color='darkgray', alpha=0.5)
    safe_fill_between(ax,dataframe_45_year.index, dataframe_45_year.astype(float).quantile(0.25,axis=1),dataframe_45_year.quantile(0.75,axis=1), color='blue', alpha=0.5)
    safe_fill_between(ax,dataframe_85_year.index, dataframe_85_year.astype(float).quantile(0.25,axis=1),dataframe_85_year.quantile(0.75,axis=1), color='red', alpha=0.5)
    
    ax.set_ylim(vmin,vmax)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax.legend(loc = 2)
    if var=='Prec':
        ax.set_ylabel('Precipitación (mm)')
    elif var=='Tmax':
        ax.set_ylabel('Temperatura máxima (ºC)')
    elif var=='Tmin':
        ax.set_ylabel('Temperatura mínima (ºC)')
    fig.savefig(path_project+'/07_INFORME/Figuras/Incertidumbre_CC_'+var+'.png',bbox_inches='tight',dpi=350)
    

def SPI(serie_pcp, verbose=False):
    """Calcular el 'standard precipitation index' (SPI) de una serie de
    precipitación
    
    Entradas:
    ---------
    serie_pcp: Series. Serie de precipitación
    verbose:   boolean. Si se muestran los coeficientes ajustados para la
               distribución gamma
    
    Salidas:
    --------
    SPIs:      Series. Serie de SPI
    """
    
    # ajustar la función de distribución gamma
    alpha, loc, beta = stats.gamma.fit(serie_pcp, floc=0)
    if verbose == True:
        print('alpha = {0:.3f}\tloc = {1:.3f}\tbeta = {2:.3f}'.format(alpha, loc,
                                                                      beta))
    
    # calcular el SPI para la serie
    SPIs = pd.Series(index=serie_pcp.index)
    for idx, pcp in zip(serie_pcp.index, serie_pcp):
        cdf = stats.gamma.cdf(pcp, alpha, loc, beta)
        SPIs[idx] = stats.norm.ppf(cdf)
        
    return SPIs



def SPI_CC(serie_pcp_hist,serie_pcp_CC, verbose=False):
    """Calcular el 'standard precipitation index' (SPI) de una serie de
    precipitación
    
    Entradas:
    ---------
    serie_pcp: Series. Serie de precipitación
    verbose:   boolean. Si se muestran los coeficientes ajustados para la
               distribución gamma
    
    Salidas:
    --------
    SPIs:      Series. Serie de SPI
    """
    
    # ajustar la función de distribución gamma
    alpha, loc, beta = stats.gamma.fit(serie_pcp_hist, floc=0)
    if verbose == True:
        print('alpha = {0:.3f}\tloc = {1:.3f}\tbeta = {2:.3f}'.format(alpha, loc,
                                                                      beta))
    
    # calcular el SPI para la serie
    SPIs = pd.Series(index=serie_pcp_CC.index)
    for idx, pcp in zip(serie_pcp_CC.index, serie_pcp_CC):
        cdf = stats.gamma.cdf(pcp, alpha, loc, beta)
        SPIs[idx] = stats.norm.ppf(cdf)
        
    return SPIs

def plot_SPI_climate_change(serie_spi_hist,serie_spi_45, serie_spi_85, title,ax,labels):
    """Crea un diagrama de línea con la evolución temporal del SPI
    
    Entradas:
    ---------
    serie_spi: Series. Serie temporal de SPI
    title:     string. Título del gráfico
    
    Salidas:
    --------
    Gráfico de línea"""
    
    # Configuración
    #fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(serie_spi_hist.index[0], serie_spi_45.index[-1]), ylim=(-3, 3))
    ax.set_title(title, fontsize=18)
    
    # hist = ax.twiny()
    # hist.spines["bottom"].set_position(("axes", -.1)) # move it down
    # #make_patch_spines_invisible(hist) 
    # make_spine_invisible(hist, "bottom")
    serie_spi_mean_hist = serie_spi_hist.median(axis=1)
    serie_spi_q25_hist  = serie_spi_hist.quantile(0.25,axis=1)
    serie_spi_q95_hist  = serie_spi_hist.quantile(0.95,axis=1)
    
    
    serie_spi_mean_rcp45 = serie_spi_45.median(axis=1)
    serie_spi_q25_rcp45  = serie_spi_45.quantile(0.25,axis=1)
    serie_spi_q95_rcp45  = serie_spi_45.quantile(0.95,axis=1)
    
    serie_spi_mean_rcp85 = serie_spi_85.median(axis=1)
    serie_spi_q25_rcp85  = serie_spi_85.quantile(0.25,axis=1)
    serie_spi_q95_rcp85  = serie_spi_85.quantile(0.95,axis=1)
    
    
    # Gráfico de línea del SPI
    ax.plot(serie_spi_mean_hist.rolling(3,min_periods=1, center=True).mean(), color='k', linewidth=1.2, label = 'Hist' )
    ax.plot(serie_spi_mean_rcp45.rolling(3,min_periods=1, center=True).mean(), color='blue', linewidth=1.2, label = labels[0] )
    ax.plot(serie_spi_mean_rcp85.rolling(3,min_periods=1, center=True).mean(), color='red', linewidth=1.2, label  = labels[1])
    
    # Fondo con la leyenda de cada rango de SPI
    ax.fill_between(serie_spi_45.index, -3, -2, color='black', alpha=0.4-0.1,
                    label='sequía extrema')
    ax.fill_between(serie_spi_45.index, -2, -1.5, color='black', alpha=0.3-0.1,
                    label='sequía severa')
    ax.fill_between(serie_spi_45.index, -1.5, -1, color='black', alpha=0.2-0.1,
                    label='sequía moderada')
    ax.fill_between(serie_spi_45.index, -1, 0, color='black', alpha=0.05,
                    label='sequía ligera')
    ax.fill_between(serie_spi_45.index, 0, 1, color='cyan', alpha=0.05,
                    label='húmedo ligero')
    ax.fill_between(serie_spi_45.index, 1, 1.5, color='cyan', alpha=0.2-0.1,
                    label='húmedo moderado')
    ax.fill_between(serie_spi_45.index, 1.5, 2, color='cyan', alpha=0.3-0.1,
                    label='húmedo severo')
    ax.fill_between(serie_spi_45.index, 2, 3, color='cyan', alpha=0.4-0.1,
                    label='húmedo extremo')
    
    ax.fill_between(serie_spi_mean_hist.index, -3, -2, color='black', alpha=0.4-0.1)
    ax.fill_between(serie_spi_mean_hist.index, -2, -1.5, color='black', alpha=0.3-0.1)
    ax.fill_between(serie_spi_mean_hist.index, -1.5, -1, color='black', alpha=0.2-0.1)
    ax.fill_between(serie_spi_mean_hist.index, -1, 0, color='black', alpha=0.05)
    ax.fill_between(serie_spi_mean_hist.index, 0, 1, color='cyan', alpha=0.05)
    ax.fill_between(serie_spi_mean_hist.index, 1, 1.5, color='cyan', alpha=0.1)
    ax.fill_between(serie_spi_mean_hist.index, 1.5, 2, color='cyan', alpha=0.2)
    ax.fill_between(serie_spi_mean_hist.index, 2, 3, color='cyan', alpha=0.3)
    
    serie_spi_hist.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='grey',  alpha=0.2, label='',ax=ax)
    serie_spi_hist.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='grey',  alpha=0.2, label='',ax=ax)
    
    serie_spi_45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax) 
    serie_spi_45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax)
    
    serie_spi_85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax) 
    serie_spi_85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax)
    
    ax.fill_between(serie_spi_mean_hist.index, serie_spi_hist.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_hist.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='grey', alpha=0.2)
    ax.fill_between(serie_spi_45.index, serie_spi_45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='blue', alpha=0.2)
    ax.fill_between(serie_spi_85.index, serie_spi_85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='red', alpha=0.2)
    
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_ylim(-3,3)
    
    ax.vlines(f'{serie_spi_45.index.year[0]}-01-01', -3, 3, 'k', linestyle = '-')
    ax.set_ylabel("SPI",fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    
    
def fig_SPI(path_project,models,climate_change):

    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']

    fig, ax = plt.subplots(figsize=(12, 7))

    pickle_in_hist_tasmax = open(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}/dict_hist.pickle","rb")
    dict_hist = pickle.load(pickle_in_hist_tasmax)

    pickle_in_CC = open(f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}/dict_CC.pickle","rb")
    dict_CC = pickle.load(pickle_in_CC)

    dataframe_hist=pd.DataFrame(index=pd.date_range(start='1950-01-01',end=f'{hist_period[1]}-12-31',freq='M'),columns=models)
    dataframe_hist.index = dataframe_hist.index.date - pd.offsets.MonthBegin(1)

    dataframe_45=pd.DataFrame(index=pd.date_range(start='2006-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_45.index = dataframe_45.index.date - pd.offsets.MonthBegin(1)

    dataframe_85=pd.DataFrame(index=pd.date_range(start=f'{fut_period[0]}-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_85.index = dataframe_85.index.date - pd.offsets.MonthBegin(1)
    for i in tqdm.tqdm(models):
        d0=dict_hist['pr'][i]
        d1=dict_CC['pr'][esce[0]][i]
        d2=dict_CC['pr'][esce[0]][i]
        d3=dict_CC['pr'][esce[0]][i]

        d4=dict_CC['pr'][esce[1]][i]
        d5=dict_CC['pr'][esce[1]][i]
        d6=dict_CC['pr'][esce[1]][i]

        d00=d0.mean(axis=1)
        d10=d1.mean(axis=1)
        d20=d2.mean(axis=1)
        d30=d3.mean(axis=1)
        d40=d4.mean(axis=1)
        d50=d5.mean(axis=1)
        d60=d6.mean(axis=1)

        d00.index = pd.to_datetime(d00.index.astype(str),errors='coerce'); d00 = d00[~d00.index.isnull()];
        d10.index = pd.to_datetime(d10.index.astype(str),errors='coerce'); d10 = d10[~d10.index.isnull()];
        d20.index = pd.to_datetime(d20.index.astype(str),errors='coerce'); d20 = d20[~d20.index.isnull()];
        d30.index = pd.to_datetime(d30.index.astype(str),errors='coerce'); d30 = d30[~d30.index.isnull()];
        d40.index = pd.to_datetime(d40.index.astype(str),errors='coerce'); d40 = d40[~d40.index.isnull()];
        d50.index = pd.to_datetime(d50.index.astype(str),errors='coerce'); d50 = d50[~d50.index.isnull()];
        d60.index = pd.to_datetime(d60.index.astype(str),errors='coerce'); d60 = d60[~d60.index.isnull()];

        dataframe_hist.loc[d00.index,i]=d00.values
        dataframe_45.loc[d10.index,i]=d10.values
        dataframe_45.loc[d20.index,i]=d20.values
        dataframe_45.loc[d30.index,i]=d30.values

        dataframe_85.loc[d40.index,i]=d40.values
        dataframe_85.loc[d50.index,i]=d50.values
        dataframe_85.loc[d60.index,i]=d60.values


    dataframe_hist_year=dataframe_hist.dropna().resample('A').sum()
    dataframe_45_year=dataframe_45.dropna().resample('A').sum()
    dataframe_45_year.index = dataframe_45_year.index - pd.offsets.YearBegin(1)
    dataframe_85_year=dataframe_85.dropna().resample('A').sum()
    dataframe_85_year.index = dataframe_85_year.index - pd.offsets.YearBegin(1)

    serie_spi_hist = pd.DataFrame(index=dataframe_hist_year.dropna().index, columns = models)
    serie_spi_45 = pd.DataFrame(index=dataframe_45_year.dropna().index, columns=models)
    serie_spi_85 = pd.DataFrame(index=dataframe_85_year.dropna().index, columns=models)

    for nmodel in models:
        serie_spi_hist.loc[:,nmodel] = SPI(dataframe_hist_year.loc[:,nmodel].astype(float), verbose=False)
        serie_spi_45.loc[:,nmodel] = SPI_CC(dataframe_hist_year.loc[:,nmodel].astype(float),dataframe_45_year.loc[:,nmodel].astype(float), verbose=False)
        serie_spi_85.loc[:,nmodel] = SPI_CC(dataframe_hist_year.loc[:,nmodel].astype(float),dataframe_85_year.loc[:,nmodel].astype(float), verbose=False)

    plot_SPI_climate_change(serie_spi_hist,serie_spi_45, serie_spi_85, 'Índice de precipitation estandarizado (SPI)',ax,label_esce)

    lines = []
    labels = []  
    for i,axx in enumerate(fig.axes[:1]):
        axLine, axLabel = axx.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)


    fig.subplots_adjust(bottom=0.1)

    fig.legend(lines, labels,           
               loc = 8,ncol=4,fontsize=12)
    fig.tight_layout(pad=7)

    fig.savefig(path_project+'/07_INFORME/Figuras/SPI_CC'+'.png',bbox_inches='tight',dpi=350)

def plot_SSFI_climate_change(serie_hist, serie_spi_45, serie_spi_85, title,ax,labels):
    """Crea un diagrama de línea con la evolución temporal del SPI
    
    Entradas:
    ---------
    serie_spi: Series. Serie temporal de SPI
    title:     string. Título del gráfico
    
    Salidas:
    --------
    Gráfico de línea"""
    
    # 1. Seleccionar los últimos N años de la serie histórica
    N = 8  # Número de años a copiar
    last_years = serie_hist.index[-1].year - N + 1
    serie_ficticia = serie_hist[serie_hist.index.year >= last_years].copy()

    # 2. Asegurar que el primer valor de la ficticia sea igual al último de la real
    serie_ficticia.iloc[0] = serie_hist.iloc[-1]  # Forzar coincidencia en el punto de unión

    # 3. Generar nuevas fechas (años futuros consecutivos)
    new_dates = pd.date_range(
        start=serie_hist.index[-2] + pd.offsets.DateOffset(years=1),  # Empieza un año después del último dato
        periods=len(serie_ficticia),
        freq='Y'  # Frecuencia anual
    )
    serie_ficticia.index = new_dates

    # 4. Graficar la serie ficticia como línea discontinua
    ax.plot(
        serie_ficticia.index,
        serie_ficticia,
        linestyle='--',
        color='gray',
        alpha=0.7,
        label='Proyección ficticia (últimos 5 años)'
    )
   

    # Configuración
    #fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(serie_hist.index[0], serie_spi_45.index[-1]), ylim=(-3, 3))
    time = pd.date_range(start=serie_hist.index[0], end=serie_spi_45.index[-1], freq='YE')
    ax.set_title(title, fontsize=18)
    
   
    
    serie_spi_mean_rcp45 = serie_spi_45.median(axis=1)
    serie_spi_q25_rcp45  = serie_spi_45.quantile(0.25,axis=1)
    serie_spi_q95_rcp45  = serie_spi_45.quantile(0.95,axis=1)
    
    serie_spi_mean_rcp85 = serie_spi_85.median(axis=1)
    serie_spi_q25_rcp85  = serie_spi_85.quantile(0.25,axis=1)
    serie_spi_q95_rcp85  = serie_spi_85.quantile(0.95,axis=1)
    
    
    # Gráfico de línea del SPI
    ax.plot(serie_hist, color='black', linewidth=1.2, label = 'Historical')
    ax.plot(serie_spi_mean_rcp45.rolling(3,min_periods=1, center=True).mean(), color='blue', linewidth=1.2, label = labels[0] )
    ax.plot(serie_spi_mean_rcp85.rolling(3,min_periods=1, center=True).mean(), color='red', linewidth=1.2, label  = labels[1])
    
    # Fondo con la leyenda de cada rango de SPI
    ax.fill_between(time, -3, -2, color='black', alpha=0.4-0.1,
                    label='sequía extrema')
    ax.fill_between(time, -2, -1.5, color='black', alpha=0.3-0.1,
                    label='sequía severa')
    ax.fill_between(time, -1.5, -1, color='black', alpha=0.2-0.1,
                    label='sequía moderada')
    ax.fill_between(time, -1, 0, color='black', alpha=0.05,
                    label='sequía ligera')
    ax.fill_between(time, 0, 1, color='cyan', alpha=0.05,
                    label='húmedo ligero')
    ax.fill_between(time, 1, 1.5, color='cyan', alpha=0.2-0.1,
                    label='húmedo moderado')
    ax.fill_between(time, 1.5, 2, color='cyan', alpha=0.3-0.1,
                    label='húmedo severo')
    ax.fill_between(time, 2, 3, color='cyan', alpha=0.4-0.1,
                    label='húmedo extremo')
    
    serie_spi_45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax) 
    serie_spi_45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax)
    
    serie_spi_85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax) 
    serie_spi_85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax)
    
    ax.fill_between(serie_spi_45.index, serie_spi_45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='blue', alpha=0.2)
    ax.fill_between(serie_spi_85.index, serie_spi_85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='red', alpha=0.2)
    
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_ylim(-3,3)
    
    ax.vlines(f'{serie_spi_45.index.year[0]}-12-31', -3, 3, 'k', linestyle = '-')
    ax.set_ylabel("SSFI",fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

def fig_SSFI(path_project, models, climate_change):
    import os
    """
    Genera la figura de SSFI usando series de aportaciones ya calculadas en CSV.
    """

    # =====================
    # Configuración
    # =====================
    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']

    # Inicializa DataFrames anuales
    dataframe_hist_year = pd.DataFrame()
    dataframe_245_year = pd.DataFrame()
    dataframe_585_year = pd.DataFrame()

    # =====================
    # Lectura modelo por modelo
    # =====================
    df_hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    df_hist = df_hist.loc[hist_period[0]:hist_period[1]]
    df_hist[df_hist <= 0] = 0.000001
    df_hist_y = df_hist.resample('A').sum()
    serie_ssfi_hist = SPI(df_hist_y.iloc[:,0].astype(float), verbose=False)
    for model in tqdm.tqdm(models):
        modelo_split = model.split("_")
       
        # === SSP2 4.5 ===
        file_245 = f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change, 'Aportaciones', esce[0], modelo_split)}.csv"
        df_245 = pd.read_csv(file_245, index_col=0, parse_dates=True)
        df_245 = df_245.loc[fut_period[0]:fut_period[1]]
        df_245[df_245 <= 0] = 0.000001
        df_245_y = df_245.resample('A').sum()
        dataframe_245_year[model] = df_245_y.iloc[:,0]

        # === SSP5 8.5 ===
        file_585 = f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change, 'Aportaciones', esce[1], modelo_split)}.csv"
        df_585 = pd.read_csv(file_585, index_col=0, parse_dates=True)
        df_585 = df_585.loc[fut_period[0]:fut_period[1]]
        df_585[df_585 <= 0] = 0.000001
        df_585_y = df_585.resample('A').sum()
        dataframe_585_year[model] = df_585_y.iloc[:,0]

    # =====================
    # Calcula SSFI
    # =====================
    serie_ssfi_245 = pd.DataFrame(index=dataframe_245_year.index, columns=models)
    serie_ssfi_585 = pd.DataFrame(index=dataframe_585_year.index, columns=models)

    for model in models:
        fut_245_values = dataframe_245_year[model].astype(float)
        fut_585_values = dataframe_585_year[model].astype(float)

        serie_ssfi_245[model] = SPI_CC(df_hist_y.loc['1995':'2014'].values, fut_245_values, verbose=False)
        serie_ssfi_585[model] = SPI_CC(df_hist_y.loc['1995':'2014'].values, fut_585_values, verbose=False)

    # =====================
    # Gráfico
    # =====================
    fig, ax = plt.subplots(figsize=(12,7))

    # Función de ploteo específica SSFI
    plot_SSFI_climate_change(serie_ssfi_hist, serie_ssfi_245, serie_ssfi_585, 'Índice de caudal estandarizado (SSFI)', ax, label_esce)

    lines = []
    labels = []  
    for i,axx in enumerate(fig.axes[:1]):
        axLine, axLabel = axx.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)



    fig.subplots_adjust(bottom=0.1)

    fig.legend(lines, labels,           
               loc = 8,ncol=4,fontsize=12)
    fig.tight_layout(pad=7)

    fig.savefig(path_project+'/07_INFORME/Figuras/SSFI_CC'+'.png',bbox_inches='tight',dpi=350)
    
    
def plot_anual_change_aport(path_project, climate_change, models):
    models = models.copy()
    # Define escenarios y periodos según CMIP6 o CORDEX
    if climate_change == 'CORDEX':
        esce = ['rcp45', 'rcp85']
        label_esce = ['RCP 4.5', 'RCP 8.5']
        hist_period = ['1976', '2005']
        fut_period = ['2006', '2100']
    elif climate_change == 'CMIP6':
        esce = ['ssp245', 'ssp585']
        label_esce = ['SSP2 4.5', 'SSP5 8.5']
        hist_period = ['1995', '2014']
        fut_period = ['2015', '2100']

    # Lee aportaciones históricas
    Aport_hist = pd.read_csv(f"{path_project}/03_APORTACIONES/Aportaciones_Sim.csv", index_col=0, parse_dates=True)
    Aport_hist = Aport_hist.loc[hist_period[0]:hist_period[1]]

    # Crea DataFrames vacíos para futuro
    date_range_fut = pd.date_range(start=f'{fut_period[0]}-01-01', end='2100-12-31', freq='M')
    Aport_fut_45 = pd.DataFrame(index=date_range_fut - pd.offsets.MonthBegin(1), columns=models)
    Aport_fut_85 = pd.DataFrame(index=date_range_fut - pd.offsets.MonthBegin(1), columns=models)

    # Rellena con cada modelo
    for nmodel in models:
        modelo_split = nmodel.split("_")
        apor_45 = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change, 'Aportaciones', esce[0], modelo_split)}.csv", index_col=0, parse_dates=True)
        apor_85 = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change, 'Aportaciones', esce[1], modelo_split)}.csv", index_col=0, parse_dates=True)

        Aport_fut_45.loc[apor_45.index, nmodel] = apor_45.values.flatten()
        Aport_fut_85.loc[apor_85.index, nmodel] = apor_85.values.flatten()

    # Calcula el cambio porcentual anual para cada escenario
    def compute_change(aport_fut, aport_hist, escenario_label, esce_name):
        # Cambio anual en porcentaje respecto a histórico medio
        change = (aport_fut.dropna().resample('Y').sum() / aport_hist.resample('Y').sum().mean()[0] - 1) * 100
        change = change.loc[:, change.mean().sort_values().index]

        # Añade columna de media
        change['MEDIANA DE LOS MODELOS'] = change.median(axis=1)

        # Prepara colores y bins
        cmap = plt.cm.bwr_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (.5, .5, .5, 1.0)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        bounds = np.arange(-100, 120, 20)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        cbar_kws = {'label': 'Porcentaje de cambio (%)'}

        # Ajusta tamaño de la figura dinámicamente
        n_models = len(change.columns)
        fig_height = np.max([4.0, n_models * 0.2])
        fig, ax = plt.subplots(figsize=(20, fig_height))

        # Plot heatmap
        sns.heatmap(change.T.astype(float), annot=False, ax=ax, cmap='bwr_r', norm=norm,
                    cbar_kws=cbar_kws, vmax=100, vmin=-100, xticklabels=change.index.year)
        ax.set_yticks(np.arange(len(change.columns)) + 0.5)
        ax.set_yticklabels(change.columns, rotation=0, fontsize=8)  # Ajusta fontsize según preferencia

        ax.set_title(f'Cambio anual en aportaciones {escenario_label}', fontsize=12, fontweight="bold")

        # Guarda figura y cierra
        plt.savefig(f"{path_project}/07_INFORME/Figuras/Cambio_Anual_Modelos_{esce_name.upper()}.png", bbox_inches='tight', dpi=350)
        # plt.close()

    # Genera figuras para ambos escenarios
    compute_change(Aport_fut_45, Aport_hist, label_esce[0], esce[0])
    compute_change(Aport_fut_85, Aport_hist, label_esce[1], esce[1])
    
def plot_cambios_aport(path_project,climate_change, models):  
    #sns.set_style("white")
    #sns.set_context("poster")
    path_climate_change = f"{path_project}/05_CAMBIO_CLIMATICO/01_CLIMA/{climate_change}_BIAS_CORRECTED/"
    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
        periodos_N  = ['2011_2040','2041_2070','2071_2100']
        labels      = ['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
        labelsize   = 20
    elif climate_change=='CMIP6':
        esce       = ['ssp245','ssp585']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']
        periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']
        labels      = ['SSP 245 2021-2040','SSP 245 2041-2060','SSP 245 2061-2080','SSP 245 2081-2100',
                       'SSP 585 2021-2040','SSP 585 2041-2060','SSP 585 2061-2080','SSP 585 2081-2100']
        labelsize = 16
    
    
    Aport_hist = pd.read_csv(f'{path_project}/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    Aport_hist[Aport_hist<=0] = 0.00000001
    Aport_hist = Aport_hist.loc[hist_period[0]:hist_period[1]]
    Aport_fut_45 = pd.DataFrame(index=pd.date_range(start=f'{fut_period[0]}-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_45.index = Aport_fut_45.index.date - pd.offsets.MonthBegin(1)
    Aport_fut_85 = pd.DataFrame(index=pd.date_range(start=f'{fut_period[0]}-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_85.index = Aport_fut_85.index.date - pd.offsets.MonthBegin(1)
    
    fig, ax = plt.subplots(nrows=len(periodos_N), ncols=2 ,figsize=(15, 16))
    col=0
    row=0
    l=0
    
    for i,rcp in enumerate(esce):
        for j, p in enumerate(periodos_N):
            factor_mean = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_q25  = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_q95  = pd.DataFrame(index=np.arange(1,13),columns=models)
            for nmod in models:
                modelo_split = nmod.split("_")
                apor_c   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',rcp,modelo_split)}.csv",index_col=0, parse_dates=True)
                apor_c   = apor_c.loc[p.split('_')[0] : p.split('_')[1]]
                
                for m in range(1,13):
                    apor_m   = apor_c[apor_c.index.month==m]
                    apor_m[apor_m<=0] = 0.00000001
                    factor_mean.loc[m,nmod] = float(apor_m.mean().values) * 100 / float(Aport_hist[Aport_hist.index.month==m].mean().values) - 100
                    factor_q25.loc[m,nmod]  = float(apor_m.quantile(0.25).values) / float(Aport_hist[Aport_hist.index.month==m].quantile(0.25).values)
                    factor_q95.loc[m,nmod]  = float(apor_m.quantile(0.95).values) / float(Aport_hist[Aport_hist.index.month==m].quantile(0.95).values)


            l1=ax[j,col].bar(np.arange(1,13),factor_mean.median(axis=1).values,color = 'dodgerblue',label='Aportación')[0]
            ax2= ax[j,col].twinx() 
            l2=ax2.plot(np.arange(1,13),factor_q95.median(axis=1).values,linestyle='-', marker='o',color='red', label='Cuantil 95%')[0]
            l3=ax2.plot(np.arange(1,13),factor_q25.median(axis=1).values,linestyle='-',marker='o', color='darkblue',label='Cuantil 25%')[0]

            ax[j,col].set_ylim(-100,100)
            ax[j,col].tick_params(axis = 'both', which = 'major', labelsize = labelsize)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = labelsize)

            ax[j,col].set_ylabel('Cambios en\n aportaciones medias (%)',fontsize = labelsize)
            ax2.set_ylabel('Cambios en cuantiles (xF)',fontsize = labelsize)
            ax2.set_yticks([-1,0,1,2,3])
            ax2.set_yticklabels(['','0','1','2','3'])
            ax2.set_ylim(-1,3)

            ax[j,col].set_title(labels[l],fontsize = 22)
            ax[j,col].set_xticks(np.arange(1,13))
            ax[j,col].grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
            l=l+1
        col=col+1
    
    # Recoger elementos únicos de leyenda usando fig.axes
    unique_legend = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in unique_legend:
                unique_legend[l] = h

    # Añadir leyenda global sin duplicados
    fig.legend(
        handles=list(unique_legend.values()),
        labels=list(unique_legend.keys()),
        loc=8,
        ncol=3,
        fontsize=labelsize)

    fig.tight_layout(pad=5)
    fig.suptitle('Cambios en el régimen medio mensual', fontsize=25,y=0.99)
    fig.savefig(path_project+'/07_INFORME/Figuras/Cambios_Aportaciones_Mensuales.png',bbox_inches='tight',dpi=350)
    
    
def analisis_aport_month(path_project,climate_change, models,period):
    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        hist_period = ['1976','2005']
    elif climate_change=='CMIP6':
        esce       = ['ssp245','ssp585']
        hist_period = ['1995','2014']
 
    df_anual_mean_hist   = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])
    df_anual_std_hist    = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])
    df_anual_Q25_hist    = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])

    df_anual_mean_45   = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_std_45    = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_Q25_45    = pd.DataFrame(index=np.arange(1,13),columns=models)

    df_anual_mean_85   = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_std_85    = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_Q25_85    = pd.DataFrame(index=np.arange(1,13),columns=models)
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc[hist_period[0]:hist_period[1]]
    SIM_Hist[SIM_Hist<=0] = 0.00000001
    
    df_anual_mean_hist.iloc[:]       = SIM_Hist.groupby(by = SIM_Hist.index.month).mean().values
    df_anual_std_hist.iloc[:]        = SIM_Hist.groupby(by = SIM_Hist.index.month).std().values
    df_anual_Q25_hist.iloc[:]        = SIM_Hist.groupby(by = SIM_Hist.index.month).quantile(0.25).values
    
    for nmod in models:
        modelo_split = nmod.split("_")
        data_45   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[0],modelo_split)}.csv",index_col=0, parse_dates=True)
        data_45   = data_45.loc[period.split('_')[0] : period.split('_')[1]]
        
        data_85   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[1],modelo_split)}.csv",index_col=0, parse_dates=True)
        data_85   = data_85.loc[period.split('_')[0] : period.split('_')[1]]
    
        df_anual_mean_45.loc[:,nmod]   = data_45.groupby(by = data_45.index.month).mean().values 
        df_anual_std_45.loc[:,nmod]    = data_45.groupby(by = data_45.index.month).std().values
        df_anual_Q25_45.loc[:,nmod]    = data_45.groupby(by = data_45.index.month).quantile(0.25).values

        df_anual_mean_85.loc[:,nmod]   = data_85.groupby(by = data_85.index.month).mean().values
        df_anual_std_85.loc[:,nmod]    = data_85.groupby(by = data_85.index.month).std().values
        df_anual_Q25_85.loc[:,nmod]    = data_85.groupby(by = data_85.index.month).quantile(0.25).values
        
    return df_anual_mean_hist, df_anual_std_hist, df_anual_Q25_hist, df_anual_mean_45, df_anual_std_45, df_anual_Q25_45, df_anual_mean_85,df_anual_std_85,df_anual_Q25_85

def plot_analisis_mensual_aport(path_project,climate_change,models):

    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
        periodos_N  = ['2011_2040','2041_2070','2071_2100']
        labels      = ['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
        labelsize   = 20
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']
        periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']
        labels      = ['SSP 245 2021-2040','SSP 245 2041-2060','SSP 245 2061-2080','SSP 245 2081-2100',
                       'SSP 585 2021-2040','SSP 585 2041-2060','SSP 585 2061-2080','SSP 585 2081-2100']



    mux_cc = pd.MultiIndex.from_product([periodos_N,label_esce,['Median','std','Q25','Q75']])
    Analisis_mensual_CC = pd.DataFrame(index = np.arange(1,13),columns=mux_cc,dtype=float)

    mux_hist = pd.MultiIndex.from_product([['Hist'],['Mean']])
    Analisis_mensual_hist = pd.DataFrame(index = np.arange(1,13),columns=mux_hist,dtype=float)

    fig, ax   = plt.subplots(nrows=len(periodos_N),ncols=3 ,figsize=(30, 28))
    axess = ax.flatten()
    n_axes = 0
    palette = {
        'Hist': 'tab:grey',
         label_esce[0]: 'tab:blue',
         label_esce[1]: 'tab:red',
    }
    sns.set(style="whitegrid")
    for i, period in enumerate(periodos_N):
        [df_month_mean_hist, df_month_std_hist, 
         df_month_Q25_hist, df_month_mean_45, 
         df_month_std_45, df_month_Q25_45, 
         df_month_mean_85,df_month_std_85,df_month_Q25_85] = analisis_aport_month(path_project,climate_change,models,period)

        Analisis_mensual_hist['Hist'].loc[:]    = df_month_mean_hist.astype(float).values

        Analisis_mensual_CC[period][label_esce[0]]['Median'][:] = df_month_mean_45.astype(float).median(axis=1).values
        Analisis_mensual_CC[period][label_esce[0]]['std'][:]    = df_month_mean_45.astype(float).std(axis=1).values
        Analisis_mensual_CC[period][label_esce[0]]['Q25'][:]    = df_month_mean_45.astype(float).quantile(0.25,axis=1).values
        Analisis_mensual_CC[period][label_esce[0]]['Q75'][:]    = df_month_mean_45.astype(float).quantile(0.75,axis=1).values


        Analisis_mensual_CC[period][label_esce[1]]['Median'][:] = df_month_mean_85.astype(float).mean(axis=1).values
        Analisis_mensual_CC[period][label_esce[1]]['std'][:]    = df_month_mean_85.astype(float).std(axis=1).values
        Analisis_mensual_CC[period][label_esce[1]]['Q25'][:]    = df_month_mean_85.astype(float).quantile(0.25,axis=1).values
        Analisis_mensual_CC[period][label_esce[1]]['Q75'][:]    = df_month_mean_45.astype(float).quantile(0.75,axis=1).values

        df1=pd.DataFrame(np.hstack((df_month_mean_hist.values.flatten(),df_month_mean_45.values.flatten(),df_month_mean_85.values.flatten())),columns=['Aport'])
        df1['Month'] = np.hstack((df_month_mean_45.index,np.repeat(df_month_mean_45.index,len(models)),np.repeat(df_month_mean_45.index,len(models))))
        df1['SCE'] = np.hstack((np.repeat('Hist',12),np.repeat([label_esce[0],label_esce[1]],len(models)*12)))

        df2=pd.DataFrame(np.hstack((df_month_std_hist.values.flatten(),df_month_std_45.values.flatten(),df_month_std_85.values.flatten())),columns=['Aport'])
        df2['Month'] = np.hstack((df_month_std_45.index,np.repeat(df_month_std_45.index,len(models)),np.repeat(df_month_std_45.index,len(models))))
        df2['SCE'] = np.hstack((np.repeat('Hist',12),np.repeat([label_esce[0],label_esce[1]],len(models)*12)))

        df3=pd.DataFrame(np.hstack((df_month_Q25_hist.values.flatten(),df_month_Q25_45.values.flatten(),df_month_Q25_85.values.flatten())),columns=['Aport'])
        df3['Month'] = np.hstack((df_month_Q25_45.index,np.repeat(df_month_Q25_45.index,len(models)),np.repeat(df_month_Q25_45.index,len(models))))
        df3['SCE'] = np.hstack((np.repeat('Hist',12),np.repeat([label_esce[0],label_esce[1]],len(models)*12)))

        g1 = sns.barplot(x='Month',y='Aport',hue="SCE", data=df1, palette=palette,ax=axess[n_axes],estimator=np.median)
        axess[n_axes].set_title('Aportaciones Mensuales Medias',fontsize=22)
        axess[n_axes].set_ylabel('Período ' +period+'\n Aportación Hm3/mes',fontsize=20)
        axess[n_axes].set_xlabel('',fontsize=15)
        axess[n_axes].tick_params(labelrotation=30,labelsize=20)
        axess[n_axes].legend(fontsize=18)

        g2 = sns.barplot(x='Month',y='Aport',hue="SCE", data=df2, palette=palette,ax=axess[n_axes+1],estimator=np.median)
        axess[n_axes+1].set_title('Desviación de aportaciones Mensuales',fontsize=22)
        axess[n_axes+1].set_ylabel('',fontsize=15)
        axess[n_axes+1].set_xlabel('',fontsize=15)
        axess[n_axes+1].tick_params(labelrotation=30,labelsize=20)
        axess[n_axes+1].legend(fontsize=18)

        g3 = sns.barplot(x='Month',y='Aport',hue="SCE", data=df3, palette=palette,ax=axess[n_axes+2],estimator=np.median)
        axess[n_axes+2].set_title('Cuantil del 25% de aportaciones Mensuales',fontsize=22)
        axess[n_axes+2].set_ylabel('',fontsize=15)
        axess[n_axes+2].set_xlabel('',fontsize=15)
        axess[n_axes+2].tick_params(labelrotation=30,labelsize=20)

        axess[n_axes+2].legend(fontsize=18)



        n_axes = n_axes + 3

    Analisis_mensual_hist.to_excel(path_project+'/06_ANALISIS_RESULTADOS/Analisis_mensual_Hist'+'.xlsx')
    Analisis_mensual_CC.to_excel(path_project+'/06_ANALISIS_RESULTADOS/Analisis_mensual_CC'+'.xlsx')
    fig.savefig(path_project+'/07_INFORME/Figuras/Analisis_Aportaciones_Mensuales.png',bbox_inches='tight',dpi=350)
    
    

def SPI(serie_pcp, verbose=False):
    """Calcular el 'standard precipitation index' (SSFI) de una serie de
    aportaciones
    
    Entradas:
    ---------
    serie_pcp: Series. Serie de aportaciones
    verbose:   boolean. Si se muestran los coeficientes ajustados para la
               distribución gamma
    
    Salidas:
    --------
    SSFIs:      Series. Serie de SSFI
    """
    
    # ajustar la función de distribución gamma
    alpha, loc, beta = stats.gamma.fit(serie_pcp, floc=0)
    if verbose == True:
        print('alpha = {0:.3f}\tloc = {1:.3f}\tbeta = {2:.3f}'.format(alpha, loc,
                                                                      beta))
    
    # calcular el SPI para la serie
    SPIs = pd.Series(index=serie_pcp.index)
    for idx, pcp in zip(serie_pcp.index, serie_pcp):
        cdf = stats.gamma.cdf(pcp, alpha, loc, beta)
        SPIs[idx] = stats.norm.ppf(cdf)
        
    return SPIs

def SPI_CC(serie_pcp_hist,serie_pcp_CC, verbose=False):
    """Calcular el 'standard precipitation index' (SPI) de una serie de
    precipitación
    
    Entradas:
    ---------
    serie_pcp: Series. Serie de precipitación
    verbose:   boolean. Si se muestran los coeficientes ajustados para la
               distribución gamma
    
    Salidas:
    --------
    SPIs:      Series. Serie de SPI
    """
    
    # ajustar la función de distribución gamma
    alpha, loc, beta = stats.gamma.fit(serie_pcp_hist, floc=0)
    if verbose == True:
        print('alpha = {0:.3f}\tloc = {1:.3f}\tbeta = {2:.3f}'.format(alpha, loc,
                                                                      beta))
    
    # calcular el SPI para la serie
    SPIs = pd.Series(index=serie_pcp_CC.index)
    for idx, pcp in zip(serie_pcp_CC.index, serie_pcp_CC):
        cdf = stats.gamma.cdf(pcp, alpha, loc, beta)
        SPIs[idx] = stats.norm.ppf(cdf)
        
    return SPIs

#### Calculate SSFI
def analysis_SSFI(path_project,climate_change,models):

    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
        periodos_N  = ['2011_2040','2041_2070','2071_2100']
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']
        periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']


    serie_spi_hist = pd.DataFrame(index=pd.date_range(start=f'{hist_period[0]}-01-01',end=f'{hist_period[1]}-12-31',freq='M'), columns=['SSFI'])
    serie_spi_hist.index = serie_spi_hist.index.date - pd.offsets.MonthBegin(1)
   
    serie_spi_hist_year = pd.DataFrame(index=pd.date_range(start=f'{hist_period[0]}-01-01',end=f'{hist_period[1]}-12-31',freq='Y'), columns=['SSFI'])
    serie_spi_hist_year.index = serie_spi_hist_year.index.date - pd.offsets.MonthBegin(1)
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc[hist_period[0]:hist_period[1]]
    SIM_Hist[SIM_Hist<=0] = 0.00000001
    SIM_Hist_year = SIM_Hist.resample('A').sum()
    
    serie_spi_hist.iloc[:,] = SPI(SIM_Hist.iloc[:,0].dropna().astype(float), verbose=False).values.reshape(-1,1)
    serie_spi_hist_year.iloc[:,] = SPI(SIM_Hist_year.iloc[:,0].dropna().astype(float), verbose=False).values.reshape(-1,1)
    
    serie_spi_hist.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_{hist_period[0]}_{hist_period[1]}.csv')
    serie_spi_hist_year.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_{hist_period[0]}_{hist_period[1]}.csv')
    
    for period in periodos_N:
        serie_spi_45 = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='M'), columns=models)
        serie_spi_45.index = serie_spi_45.index.date - pd.offsets.MonthBegin(1)
        serie_spi_85 = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='M'), columns=models)
        serie_spi_85.index = serie_spi_85.index.date - pd.offsets.MonthBegin(1)
        
        serie_spi_45_year = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='Y'), columns=models)
        serie_spi_85_year = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='Y'), columns=models)
        for nmod in models:
            modelo_split = nmod.split("_")
            data_45   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[0],modelo_split)}.csv",index_col=0, parse_dates=True)
            data_45   = data_45.loc[period.split('_')[0] : period.split('_')[1]]

            data_85   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[1],modelo_split)}.csv",index_col=0, parse_dates=True)
            data_85   = data_85.loc[period.split('_')[0] : period.split('_')[1]]
            
            data_45[data_45==0] = 0.00000000001
            data_85[data_85==0] = 0.00000000001

            data_45_year = data_45.dropna().resample('A').sum()
            data_85_year = data_85.dropna().resample('A').sum()

            serie_spi_45.loc[data_45.index,nmod] = SPI_CC(SIM_Hist.iloc[:,0],data_45.iloc[:,0].dropna().astype(float), verbose=False)
            serie_spi_85.loc[data_85.index,nmod] = SPI_CC(SIM_Hist.iloc[:,0],data_85.iloc[:,0].dropna().astype(float), verbose=False)

            serie_spi_45_year.loc[data_45_year.index,nmod] = SPI_CC(SIM_Hist_year.iloc[:,0],data_45_year.iloc[:,0].dropna().astype(float), verbose=False)
            serie_spi_85_year.loc[data_85_year.index,nmod] = SPI_CC(SIM_Hist_year.iloc[:,0],data_85_year.iloc[:,0].dropna().astype(float), verbose=False)

        serie_spi_45.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_{esce[0]}_{period}.csv')
        serie_spi_45_year.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_{esce[0]}_{period}.csv')

        serie_spi_85.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_{esce[1]}_{period}.csv')
        serie_spi_85_year.to_csv(f'{path_project}/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_{esce[1]}_{period}.csv')
        
    
def n_meses_aport(serie, value):
    # Inicializa lista vacía (no se utiliza posteriormente, podría eliminarse)
    no_aport = list()
    
    # Crea un DataFrame copia de la serie de entrada
    serie_aport = pd.DataFrame(serie.copy())
    
    # Marca con 1 los valores menores al umbral (sin aporte)
    serie_aport[serie < value] = 1
    
    # Marca con 0 los valores mayores o iguales al umbral (hay aporte)
    serie_aport[serie >= value] = 0
    
    # Crea un DataFrame de ceros con la misma estructura para contar rachas consecutivas
    ncon = pd.DataFrame(serie_aport.copy() * 0)
    
    # Recorre la serie para contar rachas consecutivas de meses sin aporte
    for i in range(len(serie_aport)):
        if serie_aport.iloc[i].values == 1:
            # Si en el mes actual no hay aporte, suma 1 al contador anterior
            ncon.iloc[i] = ncon.iloc[i-1] + 1
        else:
            # Si hay aporte, reinicia el contador a 0
            ncon.iloc[i] = 0
    
    # Crea una copia de la serie original para contar meses con aporte casi nulo (<= 1e-11)
    serie_aport_0 = serie.copy()
    
    # Marca con NaN los valores mayores a 1e-11 (hay aporte)
    serie_aport_0[serie_aport_0 > 0.00000000001] = np.nan
    
    # Marca con 1 los valores menores o iguales a 1e-11 (sin aporte prácticamente)
    serie_aport_0[serie_aport_0 <= 0.00000000001] = 1
    
    # Estas líneas parecen redundantes: intentan reemplazar valores exactos de 1e-11 por NaN
    # pero en 'serie_aport' nunca se asignó ese valor exacto en el código actual.
    serie_aport[serie_aport == 0.00000000001] = np.nan
    
    # Elimina filas con NaN en 'serie_aport' (filtrado final)
    serie_aport = serie_aport.dropna()
    
    # Obtiene los meses únicos presentes en la serie (no se usa en la función)
    months = np.unique(serie_aport.index.month)
    
    # Devuelve:
    # 1. Número de meses con aportes menores al umbral
    # 2. Racha máxima de meses consecutivos sin aporte
    # 3. Número de meses con aporte casi nulo (≤ 1e-11)
    return serie_aport.sum().values[0], ncon.max().values[0], serie_aport_0.sum()


def calculate_indicadores(path_project,name_embalse,climate_change,models):
    if climate_change=='CORDEX':
        esce        = ['rcp45','rcp85']
        label_esce  = ['RCP 4.5','RCP 8.5']
        hist_period = ['1976','2005']
        fut_period  = ['2006','2100']
        periodos_N  = ['2011_2040','2041_2070','2071_2100']
        labels      = ['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
        labelsize   = 20
    elif climate_change=='CMIP6':
        esce        = ['ssp245','ssp585']
        label_esce  = ['SSP2 4.5','SSP5 8.5']
        hist_period = ['1995','2014']
        fut_period  = ['2015','2100']
        periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']
        labels      = ['SSP 245 2021-2040','SSP 245 2041-2060','SSP 245 2061-2080','SSP 245 2081-2100',
                       'SSP 585 2021-2040','SSP 585 2041-2060','SSP 585 2061-2080','SSP 585 2081-2100']

    
    mux_cc = pd.MultiIndex.from_product([periodos_N,label_esce,['Nº de meses con aportaciones < Q25',
                                                                                                'Nº máximo de meses consecutivos con aportaciones < Q25',
                                                                                                'Nº de meses con aportaciones == 0','SSFI'],['Mean','Max','Min']])

    Analisis_mensual_CC = pd.DataFrame(index =[name_embalse],columns=mux_cc,dtype=float)

    mux_hist = pd.MultiIndex.from_product([['Hist'],['Nº de meses con aportaciones < Q25',
                                                     'Nº máximo de meses consecutivos con aportaciones < Q25',
                                                     'Nº de meses con aportaciones == 0',
                                                    'SSFI']])
    Analisis_mensual_hist = pd.DataFrame(index =  [name_embalse],columns=mux_hist,dtype=float)
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc[hist_period[0]:hist_period[1]]
    SIM_Hist[SIM_Hist==0] = 0.00000000001
    
    Analisis_mensual_hist['Hist'].iloc[0,:3] = n_meses_aport(SIM_Hist.iloc[:,0],np.percentile(SIM_Hist.values,25))
    
    Analisis_mensual_hist['Hist'].iloc[0,3]  = SPI(SIM_Hist.iloc[:,0].astype(float), verbose=False).mean()
    
   

    for i, period in enumerate(periodos_N):
        Analisis_models_45 = pd.DataFrame(index = models,columns=np.arange(0,4))
        Analisis_models_85 = pd.DataFrame(index = models,columns=np.arange(0,4))
        for nmod in models:
            modelo_split = nmod.split("_")
            data_45   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[0],modelo_split)}.csv",index_col=0, parse_dates=True)
            data_45   = data_45.loc[period.split('_')[0] : period.split('_')[1]]

            data_85   = pd.read_csv(f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',esce[1],modelo_split)}.csv",index_col=0, parse_dates=True)
            data_85   = data_85.loc[period.split('_')[0] : period.split('_')[1]]
            
            data_45[data_45==0] = 0.00000000001
            data_85[data_85==0] = 0.00000000001

            data_rcp45_year = data_45.dropna().resample('A').sum()
            data_rcp85_year = data_85.dropna().resample('A').sum()


            Analisis_models_45.loc[nmod,:2] = n_meses_aport(data_45.iloc[:,0].astype(float),np.percentile(SIM_Hist.values,25))
            Analisis_models_85.loc[nmod,:2] = n_meses_aport(data_85.iloc[:,0].astype(float),np.percentile(SIM_Hist.values,25))
            Analisis_models_45.loc[nmod,3] = SPI_CC(SIM_Hist.iloc[:,0].astype(float),data_45.iloc[:,0].dropna().astype(float), verbose=False).mean()
            Analisis_models_85.loc[nmod,3] = SPI_CC(SIM_Hist.iloc[:,0].astype(float),data_85.iloc[:,0].dropna().astype(float), verbose=False).mean()

        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_45.mean()[0])
        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_45.max()[0]
        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_45.min()[0]

        Analisis_mensual_CC[period][label_esce[0]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_45.mean()[1])
        Analisis_mensual_CC[period][label_esce[0]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_45.max()[1]
        Analisis_mensual_CC[period][label_esce[0]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_45.min()[1]

        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones == 0']['Mean'].loc[name_embalse] = int(Analisis_models_45.mean()[2])
        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones == 0']['Max'].loc[name_embalse]  = Analisis_models_45.max()[2]
        Analisis_mensual_CC[period][label_esce[0]]['Nº de meses con aportaciones == 0']['Min'].loc[name_embalse]  = Analisis_models_45.min()[2]

        Analisis_mensual_CC[period][label_esce[0]]['SSFI']['Mean'].loc[name_embalse] = Analisis_models_45.mean()[3]
        Analisis_mensual_CC[period][label_esce[0]]['SSFI']['Max'].loc[name_embalse]  = Analisis_models_45.max()[3]
        Analisis_mensual_CC[period][label_esce[0]]['SSFI']['Min'].loc[name_embalse]  = Analisis_models_45.min()[3]


        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_85.mean()[0])
        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_85.max()[0]
        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_85.min()[0]

        Analisis_mensual_CC[period][label_esce[1]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_85.mean()[1])
        Analisis_mensual_CC[period][label_esce[1]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_85.max()[1]
        Analisis_mensual_CC[period][label_esce[1]]['Nº máximo de meses consecutivos con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_85.min()[1]

        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones == 0']['Mean'].loc[name_embalse] = int(Analisis_models_85.mean()[2])
        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones == 0']['Max'].loc[name_embalse]  = Analisis_models_85.max()[2]
        Analisis_mensual_CC[period][label_esce[1]]['Nº de meses con aportaciones == 0']['Min'].loc[name_embalse]  = Analisis_models_85.min()[2]

        Analisis_mensual_CC[period][label_esce[1]]['SSFI']['Mean'].loc[name_embalse] = Analisis_models_85.mean()[3]
        Analisis_mensual_CC[period][label_esce[1]]['SSFI']['Max'].loc[name_embalse]  = Analisis_models_85.max()[3]
        Analisis_mensual_CC[period][label_esce[1]]['SSFI']['Min'].loc[name_embalse]  = Analisis_models_85.min()[3]

    Analisis_mensual_hist.to_excel(path_project+'/06_ANALISIS_RESULTADOS/'+'Indicadores_Hist.xlsx')
    Analisis_mensual_CC.to_excel(path_project+'/06_ANALISIS_RESULTADOS/'+'Indicadores_CC.xlsx')
    

class Analisis:
    """
    Clase principal para realizar el análisis de cambio climático sobre aportaciones.

    Parámetros
    ----------
    path_project : str
        Ruta al directorio principal del proyecto.
    name_project : str
        Nombre del proyecto.
    nombre_embalse : str
        Nombre del embalse analizado.
    climate_change : str, opcional
        Fuente de datos climáticos: 'CMIP6' o 'CORDEX'. Por defecto 'CMIP6'.
    logging : bool, opcional
        Si True, se mostrarán mensajes de seguimiento. Por defecto True.
    """
    def __init__(self, path_project, name_project, nombre_embalse,
                 climate_change='CMIP6', logging=True):

        self.path_project = path_project
        self.name_project = name_project
        self.nombre_embalse = nombre_embalse
        self.climate_change = climate_change.upper()
        self.logging = logging

        # Modelos climáticos según la fuente
        self.models = self._get_model_list()

    def _get_model_list(self):
        """Devuelve la lista de modelos según la fuente de cambio climático."""
        if self.climate_change == 'CORDEX':
            return [
                'CLMcom-CCLM4-8-17|CNRM-CERFACS-CNRM-CM5',
                'CLMcom-CCLM4-8-17|MOHC-HadGEM2-ES',
                'CLMcom-CCLM4-8-17|MPI-M-MPI-ESM-LR',
                'KNMI-RACMO22E|ICHEC-EC-EARTH',
                'KNMI-RACMO22E|MOHC-HadGEM2-ES',
                'MPI-CSC-REMO2009|MPI-M-MPI-ESM-LR',
                'SMHI-RCA4|CNRM-CERFACS-CNRM-CM5',
                'SMHI-RCA4|IPSL-IPSL-CM5A-MR',
                'SMHI-RCA4|MOHC-HadGEM2-ES',
                'SMHI-RCA4|MPI-M-MPI-ESM-LR'
            ]
        elif self.climate_change == 'CMIP6':
            return [
                    'CNRM-ESM2-1_r1i1p1f2_gr',
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
                    'NorESM2-MM_r1i1p1f1_gn'
                    ]
        else:
            raise ValueError(f"Fuente de cambio climático no reconocida: '{self.climate_change}'")

    def ejecutar_analisis(self):
        """Lanza todos los análisis definidos."""
        if self.logging:
            print(f"📊 Ejecutando análisis para {self.climate_change} con {len(self.models)} modelos...")

        plot_climograma(self.path_project)
        plot_clima_cuenca(self.path_project)
        plot_cambios_regimen_medio(self.path_project, self.models, self.climate_change)
        serie_climate_change(self.path_project, 'pr', self.models, self.climate_change)
        serie_climate_change(self.path_project, 'tasmax', self.models, self.climate_change)
        serie_climate_change(self.path_project, 'tasmin', self.models, self.climate_change)
        fig_SPI(self.path_project, self.models, self.climate_change)
        fig_SSFI(self.path_project, self.models, self.climate_change)
        plot_anual_change_aport(self.path_project, self.climate_change, self.models)
        plot_cambios_aport(self.path_project, self.climate_change, self.models)
        plot_analisis_mensual_aport(self.path_project, self.climate_change, self.models)
        analysis_SSFI(self.path_project,self.climate_change, self.models)
        calculate_indicadores(self.path_project,self.nombre_embalse,self.climate_change,self.models)

    def generate_fichas(self):
        import codecs
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        import shutil
        
        from datetime import date

        if self.climate_change=='CORDEX':
            esce        = ['rcp45','rcp85']
            label_esce  = ['RCP 4.5','RCP 8.5']
            hist_period = ['1976','2005']
            fut_period  = ['2006','2100']
            periodos_N  = ['2011_2040','2041_2070','2071_2100']
            labels      = ['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
            labelsize   = 20
        elif self.climate_change=='CMIP6':
            esce        = ['ssp245','ssp585']
            label_esce  = ['SSP2 4.5','SSP5 8.5']
            hist_period = ['1995','2014']
            fut_period  = ['2015','2100']
            periodos_N  = ['2021_2040','2041_2060','2061_2080','2081_2100']
            labels      = ['SSP 245 2021-2040','SSP 245 2041-2060','SSP 245 2061-2080','SSP 245 2081-2100',
                        'SSP 585 2021-2040','SSP 585 2041-2060','SSP 585 2061-2080','SSP 585 2081-2100']


        today = date.today()
        
        # shutil.copyfile(path_fichas+'/Plantilla_APORT_CC.svg',path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg')
        # shutil.copyfile(path_fichas+'/Plantilla_clima_CC.svg',path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg')
        # shutil.copyfile(path_fichas+'/Plantilla_clima_reg.svg',path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg')
        
        with open(self.path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg', 'r',encoding="utf8") as file :
            filedata = file.read()

        filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
        filedata = filedata.replace('name_embalse',self.nombre_embalse.upper())
        filedata = filedata.replace('RCP45', esce[0].upper())
        filedata = filedata.replace('RCP85', esce[1].upper())

        with codecs.open(self.path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg', 'w',encoding="utf8") as file:
            file.write(filedata)

        drawing = svg2rlg(self.path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg',resolve_entities = True)
        renderPDF.drawToFile(drawing, self.path_project+'/07_INFORME/'+'Ficha_3.pdf')   
            

        with open(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg', 'r',encoding="utf8") as file :
            filedata = file.read()

        filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
        filedata = filedata.replace('name_embalse',self.nombre_embalse.upper())

        with codecs.open(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg', 'w',encoding="utf8") as file:
            file.write(filedata)
            
            
        drawing = svg2rlg(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg',resolve_entities = True)
        renderPDF.drawToFile(drawing, self.path_project+'/07_INFORME/'+'Ficha_2.pdf')   

        with codecs.open(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg', 'r',encoding="utf8") as file :
            filedata = file.read()

        filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
        filedata = filedata.replace('name_embalse',self.nombre_embalse.upper())

        with codecs.open(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg', 'w',encoding="utf8") as file:
            file.write(filedata)
            
        drawing = svg2rlg(self.path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg',resolve_entities = True)
        renderPDF.drawToFile(drawing, self.path_project+'/07_INFORME/'+'Ficha_1.pdf')   


        
    
