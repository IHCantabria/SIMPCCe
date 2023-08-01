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
    ax.grid()
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
    
    
def plot_cambios_regimen_medio(path_project):  
    #sns.set_style("white")
    #sns.set_context("poster")
    path_climate_change = path_project+ '/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'
    periodos_N=['2011_2040','2041_2070','2071_2100']
    labels=['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
    
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
    
    prec = pd.read_csv(path_project+'/01_CLIMA/Precipitacion.csv',index_col=0, parse_dates=True).mean(axis=1).loc['1976':'2005']
    tmax = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Maxima.csv',index_col=0, parse_dates=True).mean(axis=1).loc['1976':'2005']
    tmin = pd.read_csv(path_project+'/01_CLIMA/Temperatura_Minima.csv',index_col=0, parse_dates=True).mean(axis=1).loc['1976':'2005']
    
    fig, ax = plt.subplots(nrows=3, ncols=2 ,figsize=(15, 16))
    col=0
    row=0
    l=0
    for i,rcp in enumerate(['rcp45','rcp85']):
        for j, p in enumerate(periodos_N):
            factor_prec = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_tmax = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_tmin = pd.DataFrame(index=np.arange(1,13),columns=models)
            for nmod in models:
                prec_c   =  pd.read_csv(path_climate_change+'/Prec/'+'Prec_month_CORDEX_'+rcp+'_'+nmod+'_r1i1p1.csv',index_col=0, parse_dates=True).mean(axis=1)
                tasmax_c =  pd.read_csv(path_climate_change+'/Tmax/'+'Tmax_month_CORDEX_'+rcp+'_'+nmod+'_r1i1p1.csv',index_col=0, parse_dates=True).mean(axis=1)
                tasmin_c =  pd.read_csv(path_climate_change+'/Tmin/'+'Tmin_month_CORDEX_'+rcp+'_'+nmod+'_r1i1p1.csv',index_col=0, parse_dates=True).mean(axis=1)
                
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
            ax[j,col].tick_params(axis = 'both', which = 'major', labelsize = 20)
            ax2.set_ylim(-10,10)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)

            ax[j,col].set_ylabel('Cambios en precipitación (%)',fontsize = 20)
            ax2.set_ylabel('Cambios en temperatura (ºC)',fontsize = 20)

            ax[j,col].set_title(labels[l],fontsize = 22)
            ax[j,col].set_xticks(np.arange(1,13))
            ax[j,col].grid()
            l=l+1
        col=col+1
    lines = []
    labels = []
    for i,ax in enumerate(fig.axes[::6]):
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels,           
               loc = 8,ncol=3,fontsize = 20)

    fig.tight_layout(pad=5)
    fig.suptitle('Cambios en el régimen medio mensual', fontsize=25,y=0.99)
    fig.savefig(path_project+'/07_INFORME/Figuras/Cambios_Clima.png',bbox_inches='tight',dpi=350)
    
    
def serie_climate_change(path_project,var):

    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
    
    pickle_in_hist = open(path_project+"/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/dict_hist.pickle","rb")
    dict_hist = pickle.load(pickle_in_hist)

    pickle_in_CC = open(path_project+"/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/dict_CC.pickle","rb")
    dict_CC = pickle.load(pickle_in_CC)

    dataframe_hist=pd.DataFrame(index=pd.date_range(start='1950-01-01',end='2005-12-31',freq='M'),columns=models)
    dataframe_hist.index = dataframe_hist.index.date - pd.offsets.MonthBegin(1)

    dataframe_rcp_45=pd.DataFrame(index=pd.date_range(start='2006-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_rcp_45.index = dataframe_rcp_45.index.date - pd.offsets.MonthBegin(1)

    dataframe_rcp_85=pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_rcp_85.index = dataframe_rcp_85.index.date - pd.offsets.MonthBegin(1)
    for i in tqdm.tqdm(models):
        d0=dict_hist[var][i]
        d1=dict_CC[var]['rcp45'][i]
        d2=dict_CC[var]['rcp45'][i]
        d3=dict_CC[var]['rcp45'][i]

        d4=dict_CC[var]['rcp85'][i]
        d5=dict_CC[var]['rcp85'][i]
        d6=dict_CC[var]['rcp85'][i]

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
        dataframe_rcp_45.loc[d10.index,i]=d10.values
        dataframe_rcp_45.loc[d20.index,i]=d20.values
        dataframe_rcp_45.loc[d30.index,i]=d30.values

        dataframe_rcp_85.loc[d40.index,i]=d40.values
        dataframe_rcp_85.loc[d50.index,i]=d50.values
        dataframe_rcp_85.loc[d60.index,i]=d60.values

    dataframe_hist.index=pd.to_datetime(dataframe_hist.index)
    dataframe_rcp_45.index=pd.to_datetime(dataframe_rcp_45.index)
    dataframe_rcp_85.index=pd.to_datetime(dataframe_rcp_85.index)
    
    
    
    if var=='Prec':
        dataframe_hist_year=dataframe_hist.dropna().resample('A').sum()
        dataframe_rcp_45_year=dataframe_rcp_45.dropna().resample('A').sum()
        dataframe_rcp_85_year=dataframe_rcp_85.dropna().resample('A').sum()
    else:
        dataframe_hist_year=dataframe_hist.dropna().resample('A').mean()
        dataframe_rcp_45_year=dataframe_rcp_45.dropna().resample('A').mean()
        dataframe_rcp_85_year=dataframe_rcp_85.dropna().resample('A').mean()
        
    dataframe_hist_year.index   = dataframe_hist_year.index + pd.offsets.YearBegin(1)  
    dataframe_rcp_45_year.index = dataframe_rcp_45_year.index - pd.offsets.YearBegin(1)
    dataframe_rcp_85_year.index = dataframe_rcp_85_year.index - pd.offsets.YearBegin(1)
        
        
    value_max = np.max([dataframe_hist_year.dropna().max(),dataframe_rcp_45_year.dropna().max(),dataframe_rcp_85_year.dropna().max()])
    value_min = np.min([dataframe_hist_year.dropna().min(),dataframe_rcp_45_year.dropna().min(),dataframe_rcp_85_year.dropna().min()])
    
    if var=='Prec':
        vmin = value_min-10
        vmax = value_max+10
    else:
        vmin = value_min-1.5
        vmax = value_max+1.5
        
    

    fig, ax = plt.subplots(figsize=(10, 5))
    dataframe_hist_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='darkgray',ax=ax,label='Hist')
    dataframe_rcp_45_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='blue',ax=ax,label='RCP 4.5')
    dataframe_rcp_85_year.rolling(3,min_periods=1, center=True).mean().median(axis=1).plot(kind='line',color='red',ax=ax,label='RCP 8.5')
    ax.vlines('2006-01-01', vmin, vmax, 'k', linestyle = '-', linewidth = 2)
    ax.fill_between(dataframe_hist_year.index, dataframe_hist_year.astype(float).quantile(0.25,axis=1),dataframe_hist_year.quantile(0.75,axis=1), color='darkgray', alpha=0.5)
    ax.fill_between(dataframe_rcp_45_year.index, dataframe_rcp_45_year.astype(float).quantile(0.25,axis=1),dataframe_rcp_45_year.quantile(0.75,axis=1), color='blue', alpha=0.5)
    ax.fill_between(dataframe_rcp_85_year.index, dataframe_rcp_85_year.astype(float).quantile(0.25,axis=1),dataframe_rcp_85_year.quantile(0.75,axis=1), color='red', alpha=0.5)
    ax.set_ylim(vmin,vmax)
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

def plot_SPI_climate_change(serie_spi_hist,serie_spi_rcp45, serie_spi_rcp85, title,ax):
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
    ax.set(xlim=(serie_spi_hist.index[0], serie_spi_rcp45.index[-1]), ylim=(-3, 3))
    ax.set_title(title, fontsize=18)
    
    # hist = ax.twiny()
    # hist.spines["bottom"].set_position(("axes", -.1)) # move it down
    # #make_patch_spines_invisible(hist) 
    # make_spine_invisible(hist, "bottom")
    serie_spi_mean_hist = serie_spi_hist.median(axis=1)
    serie_spi_q25_hist  = serie_spi_hist.quantile(0.25,axis=1)
    serie_spi_q95_hist  = serie_spi_hist.quantile(0.95,axis=1)
    
    
    serie_spi_mean_rcp45 = serie_spi_rcp45.median(axis=1)
    serie_spi_q25_rcp45  = serie_spi_rcp45.quantile(0.25,axis=1)
    serie_spi_q95_rcp45  = serie_spi_rcp45.quantile(0.95,axis=1)
    
    serie_spi_mean_rcp85 = serie_spi_rcp85.median(axis=1)
    serie_spi_q25_rcp85  = serie_spi_rcp85.quantile(0.25,axis=1)
    serie_spi_q95_rcp85  = serie_spi_rcp85.quantile(0.95,axis=1)
    
    
    # Gráfico de línea del SPI
    ax.plot(serie_spi_mean_hist.rolling(3,min_periods=1, center=True).mean(), color='k', linewidth=1.2, label = 'Hist' )
    ax.plot(serie_spi_mean_rcp45.rolling(3,min_periods=1, center=True).mean(), color='blue', linewidth=1.2, label = 'RCP 4.5' )
    ax.plot(serie_spi_mean_rcp85.rolling(3,min_periods=1, center=True).mean(), color='red', linewidth=1.2, label = 'RCP 8.5')
    
    # Fondo con la leyenda de cada rango de SPI
    ax.fill_between(serie_spi_rcp45.index, -3, -2, color='black', alpha=0.4-0.1,
                    label='sequía extrema')
    ax.fill_between(serie_spi_rcp45.index, -2, -1.5, color='black', alpha=0.3-0.1,
                    label='sequía severa')
    ax.fill_between(serie_spi_rcp45.index, -1.5, -1, color='black', alpha=0.2-0.1,
                    label='sequía moderada')
    ax.fill_between(serie_spi_rcp45.index, -1, 0, color='black', alpha=0.05,
                    label='sequía ligera')
    ax.fill_between(serie_spi_rcp45.index, 0, 1, color='cyan', alpha=0.05,
                    label='húmedo ligero')
    ax.fill_between(serie_spi_rcp45.index, 1, 1.5, color='cyan', alpha=0.2-0.1,
                    label='húmedo moderado')
    ax.fill_between(serie_spi_rcp45.index, 1.5, 2, color='cyan', alpha=0.3-0.1,
                    label='húmedo severo')
    ax.fill_between(serie_spi_rcp45.index, 2, 3, color='cyan', alpha=0.4-0.1,
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
    
    serie_spi_rcp45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax) 
    serie_spi_rcp45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='blue',  alpha=0.2, label='',ax=ax)
    
    serie_spi_rcp85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax) 
    serie_spi_rcp85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean().plot(linestyle = '--', color='red',  alpha=0.2, label='',ax=ax)
    
    ax.fill_between(serie_spi_mean_hist.index, serie_spi_hist.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_hist.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='grey', alpha=0.2)
    ax.fill_between(serie_spi_rcp45.index, serie_spi_rcp45.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_rcp45.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='blue', alpha=0.2)
    ax.fill_between(serie_spi_rcp85.index, serie_spi_rcp85.quantile(0.25,axis=1).rolling(3,min_periods=1, center=True).mean(),
                    serie_spi_rcp85.quantile(0.75,axis=1).rolling(3,min_periods=1, center=True).mean(), color='red', alpha=0.2)
    
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_ylim(-3,3)
    
    ax.vlines('2006-01-01', -3, 3, 'k', linestyle = '-')
    ax.set_ylabel("SPI",fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    
    
def fig_SPI(path_project):
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
                  'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
                  'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
                  'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
                  'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']

    fig, ax = plt.subplots(figsize=(12, 7))

    pickle_in_hist_tasmax = open(path_project+"/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/dict_hist.pickle","rb")
    dict_hist = pickle.load(pickle_in_hist_tasmax)

    pickle_in_CC = open(path_project+"/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/dict_CC.pickle","rb")
    dict_CC = pickle.load(pickle_in_CC)

    dataframe_hist=pd.DataFrame(index=pd.date_range(start='1950-01-01',end='2005-12-31',freq='M'),columns=models)
    dataframe_hist.index = dataframe_hist.index.date - pd.offsets.MonthBegin(1)

    dataframe_rcp_45=pd.DataFrame(index=pd.date_range(start='2006-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_rcp_45.index = dataframe_rcp_45.index.date - pd.offsets.MonthBegin(1)

    dataframe_rcp_85=pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns=models)
    dataframe_rcp_85.index = dataframe_rcp_85.index.date - pd.offsets.MonthBegin(1)
    for i in tqdm.tqdm(models):
        d0=dict_hist['Prec'][i]
        d1=dict_CC['Prec']['rcp45'][i]
        d2=dict_CC['Prec']['rcp45'][i]
        d3=dict_CC['Prec']['rcp45'][i]

        d4=dict_CC['Prec']['rcp85'][i]
        d5=dict_CC['Prec']['rcp85'][i]
        d6=dict_CC['Prec']['rcp85'][i]

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
        dataframe_rcp_45.loc[d10.index,i]=d10.values
        dataframe_rcp_45.loc[d20.index,i]=d20.values
        dataframe_rcp_45.loc[d30.index,i]=d30.values

        dataframe_rcp_85.loc[d40.index,i]=d40.values
        dataframe_rcp_85.loc[d50.index,i]=d50.values
        dataframe_rcp_85.loc[d60.index,i]=d60.values


    dataframe_hist_year=dataframe_hist.dropna().resample('A').sum()
    dataframe_rcp_45_year=dataframe_rcp_45.dropna().resample('A').sum()
    dataframe_rcp_45_year.index = dataframe_rcp_45_year.index - pd.offsets.YearBegin(1)
    dataframe_rcp_85_year=dataframe_rcp_85.dropna().resample('A').sum()
    dataframe_rcp_85_year.index = dataframe_rcp_85_year.index - pd.offsets.YearBegin(1)

    serie_spi_hist = pd.DataFrame(index=dataframe_hist_year.dropna().index, columns = models)
    serie_spi_rcp45 = pd.DataFrame(index=dataframe_rcp_45_year.dropna().index, columns=models)
    serie_spi_rcp85 = pd.DataFrame(index=dataframe_rcp_85_year.dropna().index, columns=models)

    for nmodel in models:
        serie_spi_hist.loc[:,nmodel] = SPI(dataframe_hist_year.loc[:,nmodel].astype(float), verbose=False)
        serie_spi_rcp45.loc[:,nmodel] = SPI_CC(dataframe_hist_year.loc[:,nmodel].astype(float),dataframe_rcp_45_year.loc[:,nmodel].astype(float), verbose=False)
        serie_spi_rcp85.loc[:,nmodel] = SPI_CC(dataframe_hist_year.loc[:,nmodel].astype(float),dataframe_rcp_85_year.loc[:,nmodel].astype(float), verbose=False)

    plot_SPI_climate_change(serie_spi_hist,serie_spi_rcp45, serie_spi_rcp85, 'Índice de precipitation estandarizado (SPI)',ax)

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
    
    
def plot_anual_change_aport(path_project):
    Aport_hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    Aport_hist = Aport_hist.loc['1976':'2005']
    Aport_fut_rcp45 = pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_rcp45.index = Aport_fut_rcp45.index.date - pd.offsets.MonthBegin(1)
    Aport_fut_rcp85 = pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_rcp85.index = Aport_fut_rcp85.index.date - pd.offsets.MonthBegin(1)

    for nmodel in models:
        apor_rcp45 = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmodel+'.RCP_45.csv',index_col=0, parse_dates=True)
        apor_rcp85 = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmodel+'.RCP_85.csv',index_col=0, parse_dates=True)

        Aport_fut_rcp45.loc[apor_rcp45.index,nmodel] = apor_rcp45.values.flatten()
        Aport_fut_rcp85.loc[apor_rcp85.index,nmodel] = apor_rcp85.values.flatten()
        
    change_rcp45 = (Aport_fut_rcp45.dropna().resample('Y').sum()/Aport_hist.loc['1976':'2005'].resample('Y').sum().mean()[0]-1)*100
    change_rcp45 = change_rcp45.loc[:,change_rcp45.mean().sort_values().index]
    change_rcp45['MEDIA DE LOS MODELOS'] = change_rcp45.mean(axis=1)
    
    cmap = plt.cm.bwr_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.arange(-100, 120, 20)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(20,4))
    cbar_kws={'label': 'Porcentaje de cambio (%)'}
    sns.heatmap(change_rcp45.T.astype(float), annot=False,ax=ax,cmap='bwr_r',norm=norm,cbar_kws=cbar_kws,vmax=100,vmin=-100,xticklabels=change_rcp45.index.year)
    ax.set_title('Cambio anual en aportaciones RCP 4.5',fontsize=12,fontweight="bold")
    plt.savefig(path_project+'/07_INFORME/Figuras/Cambio_Anual_Modelos_RCP45.png',bbox_inches='tight',dpi=350)
    
    change_rcp85 = (Aport_fut_rcp85.dropna().resample('Y').sum()/Aport_hist.loc['1976':'2005'].resample('Y').sum().mean()[0]-1)*100
    change_rcp85 = change_rcp85.loc[:,change_rcp85.mean().sort_values().index]
    change_rcp85['MEDIA DE LOS MODELOS'] = change_rcp85.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(20,4))
    cbar_kws={'label': 'Porcentaje de cambio (%)'}
    sns.heatmap(change_rcp85.T.astype(float), annot=False,ax=ax,cmap='bwr_r',norm=norm,cbar_kws=cbar_kws,vmax=100,vmin=-100,xticklabels=change_rcp45.index.year)
    ax.set_title('Cambio anual en aportaciones RCP 8.5',fontsize=12,fontweight="bold")
    plt.savefig(path_project+'/07_INFORME/Figuras/Cambio_Anual_Modelos_RCP85.png',bbox_inches='tight',dpi=350)
    
    
    
def plot_cambios_aport(path_project):  
    #sns.set_style("white")
    #sns.set_context("poster")
    path_climate_change = path_project+ '/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/'
    periodos_N=['2011_2040','2041_2070','2071_2100']
    labels=['RCP 45 2011-2040','RCP 45 2041-2070','RCP 45 2071-2100','RCP 85 2011-2040','RCP 85 2041-2070','RCP 85 2071-2100']
    
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
    
    Aport_hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    Aport_hist[Aport_hist<=0] = 0.00000001
    Aport_hist = Aport_hist.loc['1976':'2005']
    Aport_fut_rcp45 = pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_rcp45.index = Aport_fut_rcp45.index.date - pd.offsets.MonthBegin(1)
    Aport_fut_rcp85 = pd.DataFrame(index=pd.date_range(start='2005-01-01',end='2100-12-31',freq='M'),columns = models)
    Aport_fut_rcp85.index = Aport_fut_rcp85.index.date - pd.offsets.MonthBegin(1)
    
    fig, ax = plt.subplots(nrows=3, ncols=2 ,figsize=(15, 16))
    col=0
    row=0
    l=0
    
    for i,rcp in enumerate(['RCP_45','RCP_85']):
        for j, p in enumerate(periodos_N):
            factor_mean = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_q25  = pd.DataFrame(index=np.arange(1,13),columns=models)
            factor_q95  = pd.DataFrame(index=np.arange(1,13),columns=models)
            for nmod in models:
                apor_c   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+rcp+'.csv',index_col=0, parse_dates=True)
                apor_c   = apor_c.loc[p.split('_')[0] : p.split('_')[1]]
                
                for m in range(1,13):
                    apor_m   = apor_c[apor_c.index.month==m]
                    apor_m[apor_m<=0] = 0.00000001
                    factor_mean.loc[m,nmod] = apor_m.mean().values*100/Aport_hist[Aport_hist.index.month==m].mean().values-100
                    factor_q25.loc[m,nmod]  = apor_m.quantile(0.25).values/ Aport_hist[Aport_hist.index.month==m].quantile(0.25).values
                    factor_q95.loc[m,nmod]  = apor_m.quantile(0.95).values/ Aport_hist[Aport_hist.index.month==m].quantile(0.95).values

            l1=ax[j,col].bar(np.arange(1,13),factor_mean.median(axis=1).values,color = 'dodgerblue',label='Aportación')[0]
            ax2= ax[j,col].twinx() 
            l2=ax2.plot(np.arange(1,13),factor_q95.median(axis=1).values,linestyle='-', marker='o',color='red', label='Cuantil 95%')[0]
            l3=ax2.plot(np.arange(1,13),factor_q25.median(axis=1).values,linestyle='-',marker='o', color='darkblue',label='Cuantil 25%')[0]

            ax[j,col].set_ylim(-100,100)
            ax[j,col].tick_params(axis = 'both', which = 'major', labelsize = 20)
            ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)

            ax[j,col].set_ylabel('Cambios en\n aportaciones medias (%)',fontsize = 20)
            ax2.set_ylabel('Cambios en cuantiles (xF)',fontsize = 20)
            ax2.set_yticks([-1,0,1,2,3])
            ax2.set_yticklabels(['','0','1','2','3'])
            ax2.set_ylim(-1,3)

            ax[j,col].set_title(labels[l],fontsize = 22)
            ax[j,col].set_xticks(np.arange(1,13))
            ax[j,col].grid()
            l=l+1
        col=col+1
    lines = []
    labels = []
    for i,ax in enumerate(fig.axes[::6]):
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels,           
               loc = 8,ncol=3,fontsize = 20)

    fig.tight_layout(pad=5)
    fig.suptitle('Cambios en el régimen medio mensual', fontsize=25,y=0.99)
    fig.savefig(path_project+'/07_INFORME/Figuras/Cambios_Aportaciones_Mensuales.png',bbox_inches='tight',dpi=350)
    
    
def analisis_aport_month(path_project,period):
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
    
    
    df_anual_mean_hist   = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])
    df_anual_var_hist = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])
    df_anual_Q25_hist    = pd.DataFrame(index=np.arange(1,13),columns=['Hist'])

    df_anual_mean_rcp45   = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_var_rcp45 = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_Q25_rcp45    = pd.DataFrame(index=np.arange(1,13),columns=models)

    df_anual_mean_rcp85   = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_var_rcp85 = pd.DataFrame(index=np.arange(1,13),columns=models)
    df_anual_Q25_rcp85    = pd.DataFrame(index=np.arange(1,13),columns=models)
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc['1976':'2005']
    SIM_Hist[SIM_Hist<=0] = 0.00000001
    
    df_anual_mean_hist.iloc[:]       = SIM_Hist.groupby(by = SIM_Hist.index.month).mean().values
    df_anual_var_hist.iloc[:]        = SIM_Hist.groupby(by = SIM_Hist.index.month).var().values
    df_anual_Q25_hist.iloc[:]        = SIM_Hist.groupby(by = SIM_Hist.index.month).quantile(0.25).values
    
    for nmod in models:
        data_rcp45   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_45'+'.csv',index_col=0, parse_dates=True)
        data_rcp45   = data_rcp45.loc[period.split('_')[0] : period.split('_')[1]]
        
        data_rcp85   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_85'+'.csv',index_col=0, parse_dates=True)
        data_rcp85   = data_rcp85.loc[period.split('_')[0] : period.split('_')[1]]
    
        df_anual_mean_rcp45.loc[:,nmod]   = data_rcp45.groupby(by = data_rcp45.index.month).mean().values 
        df_anual_var_rcp45.loc[:,nmod]    = data_rcp45.groupby(by = data_rcp45.index.month).var().values
        df_anual_Q25_rcp45.loc[:,nmod]    = data_rcp45.groupby(by = data_rcp45.index.month).quantile(0.25).values

        df_anual_mean_rcp85.loc[:,nmod]   = data_rcp85.groupby(by = data_rcp85.index.month).mean().values
        df_anual_var_rcp85.loc[:,nmod]    = data_rcp85.groupby(by = data_rcp85.index.month).var().values
        df_anual_Q25_rcp85.loc[:,nmod]    = data_rcp85.groupby(by = data_rcp85.index.month).quantile(0.25).values
        
    return df_anual_mean_hist, df_anual_var_hist, df_anual_Q25_hist, df_anual_mean_rcp45, df_anual_var_rcp45, df_anual_Q25_rcp45, df_anual_mean_rcp85,df_anual_var_rcp85,df_anual_Q25_rcp85

def plot_analisis_mensual_aport(path_project):
    mux_cc = pd.MultiIndex.from_product([['2011_2040','2041_2070','2071_2100'],['RCP_45','RCP_85'],['Mean','var','Q25']])
    Analisis_mensual_CC = pd.DataFrame(index = np.arange(1,13),columns=mux_cc,dtype=float)

    mux_hist = pd.MultiIndex.from_product([['Hist'],['Mean']])
    Analisis_mensual_hist = pd.DataFrame(index = np.arange(1,13),columns=mux_hist,dtype=float)

    fig, ax   = plt.subplots(nrows=3,ncols=3 ,figsize=(30, 28))
    axess = ax.flatten()
    n_axes = 0
    palette = {
        'Hist': 'tab:grey',
        'RCP_45': 'tab:blue',
        'RCP_85': 'tab:red',
    }
    sns.set(style="whitegrid")
    for i, period in enumerate(['2011_2040','2041_2070','2071_2100']):
        [df_month_mean_hist, df_month_var_hist, 
         df_month_Q25_hist, df_month_mean_rcp45, 
         df_month_var_rcp45, df_month_Q25_rcp45, 
         df_month_mean_rcp85,df_month_var_rcp85,df_month_Q25_rcp85] = analisis_aport_month(path_project,period)

        Analisis_mensual_hist['Hist'].loc[:]    = df_month_mean_hist.astype(float).values

        Analisis_mensual_CC[period]['RCP_45']['Mean'][:]   = df_month_mean_rcp45.astype(float).mean(axis=1).values
        Analisis_mensual_CC[period]['RCP_45']['var'][:] = df_month_mean_rcp45.astype(float).var(axis=1).values
        Analisis_mensual_CC[period]['RCP_45']['Q25'][:]    = df_month_mean_rcp45.astype(float).quantile(0.25,axis=1).values

        Analisis_mensual_CC[period]['RCP_85']['Mean'][:]   = df_month_mean_rcp85.astype(float).mean(axis=1).values
        Analisis_mensual_CC[period]['RCP_85']['var'][:] = df_month_mean_rcp85.astype(float).var(axis=1).values
        Analisis_mensual_CC[period]['RCP_85']['Q25'][:]    = df_month_mean_rcp85.astype(float).quantile(0.25,axis=1).values

        df1=pd.DataFrame(np.hstack((df_month_mean_hist.values.flatten(),df_month_mean_rcp45.values.flatten(),df_month_mean_rcp85.values.flatten())),columns=['Aport'])
        df1['Month'] = np.hstack((df_month_mean_rcp45.index,np.repeat(df_month_mean_rcp45.index,10),np.repeat(df_month_mean_rcp45.index,10)))
        df1['RCP'] = np.hstack((np.repeat('Hist',12),np.repeat(['RCP_45','RCP_85'],120)))

        df2=pd.DataFrame(np.hstack((df_month_var_hist.values.flatten(),df_month_var_rcp45.values.flatten(),df_month_var_rcp85.values.flatten())),columns=['Aport'])
        df2['Month'] = np.hstack((df_month_var_rcp45.index,np.repeat(df_month_var_rcp45.index,10),np.repeat(df_month_var_rcp45.index,10)))
        df2['RCP'] = np.hstack((np.repeat('Hist',12),np.repeat(['RCP_45','RCP_85'],120)))

        df3=pd.DataFrame(np.hstack((df_month_Q25_hist.values.flatten(),df_month_Q25_rcp45.values.flatten(),df_month_Q25_rcp85.values.flatten())),columns=['Aport'])
        df3['Month'] = np.hstack((df_month_Q25_rcp45.index,np.repeat(df_month_Q25_rcp45.index,10),np.repeat(df_month_Q25_rcp45.index,10)))
        df3['RCP'] = np.hstack((np.repeat('Hist',12),np.repeat(['RCP_45','RCP_85'],120)))


        g1 = sns.barplot(x='Month',y='Aport',hue="RCP", data=df1, palette=palette,ax=axess[n_axes])
        axess[n_axes].set_title('Aportaciones Mensuales Medias',fontsize=22)
        axess[n_axes].set_ylabel('Período ' +period+'\n Aportación Hm3/mes',fontsize=20)
        axess[n_axes].set_xlabel('',fontsize=15)
        axess[n_axes].tick_params(labelrotation=30,labelsize=20)
        axess[n_axes].legend(fontsize=18)



        g2 = sns.barplot(x='Month',y='Aport',hue="RCP", data=df2, palette=palette,ax=axess[n_axes+1])
        axess[n_axes+1].set_title('Varianza de aportaciones Mensuales',fontsize=22)
        axess[n_axes+1].set_ylabel('',fontsize=15)
        axess[n_axes+1].set_xlabel('',fontsize=15)
        axess[n_axes+1].tick_params(labelrotation=30,labelsize=20)
        axess[n_axes+1].legend(fontsize=18)




        g3 = sns.barplot(x='Month',y='Aport',hue="RCP", data=df3, palette=palette,ax=axess[n_axes+2])
        axess[n_axes+2].set_title('Cuantil del 25% de aportaciones Mensuales',fontsize=22)
        axess[n_axes+2].set_ylabel('',fontsize=15)
        axess[n_axes+2].set_xlabel('',fontsize=15)
        axess[n_axes+2].tick_params(labelrotation=30,labelsize=20)

        axess[n_axes+2].legend(fontsize=18)



        n_axes = n_axes + 3

    Analisis_mensual_hist.to_excel(path_project+'/06_ANALISIS_RESULTADOS/Analisis_mensual_Hist'+'.xlsx')
    Analisis_mensual_CC.to_excel(path_project+'/06_ANALISIS_RESULTADOS/Analisis_mensual_CC'+'.xlsx')
    fig.savefig(path_project+'/07_INFORME/Figuras/Analisis_Aportaciones_Mensuales.png',bbox_inches='tight',dpi=350)
    
    
def generate_fichas(path_project,name):
    import codecs
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    import shutil
    
    from datetime import date

    today = date.today()
    
    # shutil.copyfile(path_fichas+'/Plantilla_APORT_CC.svg',path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg')
    # shutil.copyfile(path_fichas+'/Plantilla_clima_CC.svg',path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg')
    # shutil.copyfile(path_fichas+'/Plantilla_clima_reg.svg',path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg')
    
    with open(path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg', 'r',encoding="utf8") as file :
        filedata = file.read()

    filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
    filedata = filedata.replace('name_embalse',name.upper())

    with codecs.open(path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg', 'w',encoding="utf8") as file:
        file.write(filedata)
        
    drawing = svg2rlg(path_project+'/07_INFORME/Figuras/Plantilla_APORT_CC.svg',resolve_entities = True)
    renderPDF.drawToFile(drawing, path_project+'/07_INFORME/'+'Ficha_3.pdf')   
        

    with open(path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg', 'r',encoding="utf8") as file :
        filedata = file.read()

    filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
    filedata = filedata.replace('name_embalse',name.upper())

    with codecs.open(path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg', 'w',encoding="utf8") as file:
        file.write(filedata)
        
        
    drawing = svg2rlg(path_project+'/07_INFORME/Figuras/Plantilla_clima_CC.svg',resolve_entities = True)
    renderPDF.drawToFile(drawing, path_project+'/07_INFORME/'+'Ficha_2.pdf')   

    with codecs.open(path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg', 'r',encoding="utf8") as file :
        filedata = file.read()

    filedata = filedata.replace('17/05/2022', today.strftime('%d/%m/%Y'))
    filedata = filedata.replace('name_embalse',name.upper())

    with codecs.open(path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg', 'w',encoding="utf8") as file:
        file.write(filedata)
        
    drawing = svg2rlg(path_project+'/07_INFORME/Figuras/Plantilla_clima_reg.svg',resolve_entities = True)
    renderPDF.drawToFile(drawing, path_project+'/07_INFORME/'+'Ficha_1.pdf')   
    
    

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
def analysis_SSFI(path_project):
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']

    serie_spi_hist = pd.DataFrame(index=pd.date_range(start='1976-01-01',end='2005-12-31',freq='M'), columns=['SSFI'])
    serie_spi_hist.index = serie_spi_hist.index.date - pd.offsets.MonthBegin(1)
   
    
    serie_spi_hist_year = pd.DataFrame(index=pd.date_range(start='1976-01-01',end='2005-12-31',freq='Y'), columns=['SSFI'])
    serie_spi_hist_year.index = serie_spi_hist_year.index.date - pd.offsets.MonthBegin(1)
    
    
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc['1976':'2005']
    SIM_Hist[SIM_Hist<=0] = 0.00000001
    SIM_Hist_year = SIM_Hist.resample('A').sum()
    
    serie_spi_hist.iloc[:,] = SPI(SIM_Hist.iloc[:,0].dropna().astype(float), verbose=False).values.reshape(-1,1)
    serie_spi_hist_year.iloc[:,] = SPI(SIM_Hist_year.iloc[:,0].dropna().astype(float), verbose=False).values.reshape(-1,1)
    
    serie_spi_hist.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_1976_2005.csv')
    serie_spi_hist_year.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_1976_2005.csv')
    
    for period in ['2011_2040','2041_2070','2071_2100']:
        serie_spi_rcp45 = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='M'), columns=models)
        serie_spi_rcp45.index = serie_spi_rcp45.index.date - pd.offsets.MonthBegin(1)
        serie_spi_rcp85 = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='M'), columns=models)
        serie_spi_rcp85.index = serie_spi_rcp85.index.date - pd.offsets.MonthBegin(1)
        
        serie_spi_rcp45_year = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='Y'), columns=models)
        serie_spi_rcp85_year = pd.DataFrame(index=pd.date_range(start=str(period.split('_')[0])+'-01-01',end= str(period.split('_')[1]+'-12-31'),freq='Y'), columns=models)
        for nmod in models:
            data_rcp45   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_45'+'.csv',index_col=0, parse_dates=True)
            data_rcp45   = data_rcp45.loc[period.split('_')[0] : period.split('_')[1]]

            data_rcp85   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_85'+'.csv',index_col=0, parse_dates=True)
            data_rcp85   = data_rcp85.loc[period.split('_')[0] : period.split('_')[1]]
            
            data_rcp45[data_rcp45==0] = 0.00000000001
            data_rcp85[data_rcp85==0] = 0.00000000001

            data_rcp45_year = data_rcp45.dropna().resample('A').sum()
            data_rcp85_year = data_rcp85.dropna().resample('A').sum()

            serie_spi_rcp45.loc[data_rcp45.index,nmod] = SPI_CC(SIM_Hist.iloc[:,0],data_rcp45.iloc[:,0].dropna().astype(float), verbose=False)
            serie_spi_rcp85.loc[data_rcp85.index,nmod] = SPI_CC(SIM_Hist.iloc[:,0],data_rcp85.iloc[:,0].dropna().astype(float), verbose=False)

            serie_spi_rcp45_year.loc[data_rcp45_year.index,nmod] = SPI_CC(SIM_Hist_year.iloc[:,0],data_rcp45_year.iloc[:,0].dropna().astype(float), verbose=False)
            serie_spi_rcp85_year.loc[data_rcp85_year.index,nmod] = SPI_CC(SIM_Hist_year.iloc[:,0],data_rcp85_year.iloc[:,0].dropna().astype(float), verbose=False)

        serie_spi_rcp45.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_'+'RCP_45_'+period+'.csv')
        serie_spi_rcp45_year.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_'+'RCP_45_'+period+'.csv')

        serie_spi_rcp85.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Mensual_'+'RCP_85_'+period+'.csv')
        serie_spi_rcp85_year.to_csv(path_project+'/06_ANALISIS_RESULTADOS/INDICE_SSFI_Anual_'+'RCP_85_'+period+'.csv')
        
        
        
def n_meses_aport(serie,value):
    no_aport=list()
    serie_aport=pd.DataFrame(serie.copy())
    serie_aport[serie<value]=1
    serie_aport[serie>=value]=0
    ncon=pd.DataFrame(serie_aport.copy()*0)
    for i in (range(len(serie_aport))):
        if serie_aport.iloc[i].values == 1:
             ncon.iloc[i] = ncon.iloc[i-1] + 1
        else:
            ncon.iloc[i] = 0
            
    serie_aport_0=serie.copy()
    serie_aport_0[serie_aport_0>0.00000000001]=np.nan
    serie_aport_0[serie_aport_0<=0.00000000001]=1
    
    serie_aport[serie_aport==0.00000000001] = np.nan
    serie_aport = serie_aport.dropna()
    
    months = np.unique(serie_aport.index.month)

    return serie_aport.sum().values[0], ncon.max().values[0], serie_aport_0.sum()


def calculate_indicadores(path_project,name_embalse):
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5','CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
              'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR','KNMI-RACMO22E_ICHEC-EC-EARTH', 
              'KNMI-RACMO22E_MOHC-HadGEM2-ES','MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
              'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
              'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']

    
    mux_cc = pd.MultiIndex.from_product([['2011_2040','2041_2070','2071_2100'],['RCP_45','RCP_85'],['Nº de meses con aportaciones < Q25',
                                                                                                'Nº de meses consecutivos con aportaciones < Q25',
                                                                                                'Nº meses con aportaciones == 0','SSFI'],['Mean','Max','Min']])

    Analisis_mensual_CC = pd.DataFrame(index =[name_embalse],columns=mux_cc,dtype=float)

    mux_hist = pd.MultiIndex.from_product([['Hist'],['Nº de meses con aportaciones < Q25',
                                                     'Nº máximo de meses consecutivos con aportaciones < Q25',
                                                     'Nº de meses con aportaciones == 0',
                                                    'SSFI']])
    Analisis_mensual_hist = pd.DataFrame(index =  [name_embalse],columns=mux_hist,dtype=float)
    
    SIM_Hist = pd.read_csv(path_project+'/03_APORTACIONES/Aportaciones_Sim.csv',index_col=0, parse_dates=True)
    SIM_Hist = SIM_Hist.loc['1975':'2005']
    SIM_Hist[SIM_Hist==0] = 0.00000000001
    
    Analisis_mensual_hist['Hist'].iloc[0,:3] = n_meses_aport(SIM_Hist.iloc[:,0],np.percentile(SIM_Hist.values,25))
    
    Analisis_mensual_hist['Hist'].iloc[0,3]  = SPI(SIM_Hist.iloc[:,0].astype(float), verbose=False).mean()
    
   

    for i, period in enumerate(['2011_2040','2041_2070','2071_2100']):
        Analisis_models_rcp45 = pd.DataFrame(index = models,columns=np.arange(0,4))
        Analisis_models_rcp85 = pd.DataFrame(index = models,columns=np.arange(0,4))
        for nmod in models:
            
            data_rcp45   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_45'+'.csv',index_col=0, parse_dates=True)
            data_rcp45   = data_rcp45.loc[period.split('_')[0] : period.split('_')[1]]

            data_rcp85   = pd.read_csv(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/Aportaciones.'+nmod+'.'+'RCP_85'+'.csv',index_col=0, parse_dates=True)
            data_rcp85   = data_rcp85.loc[period.split('_')[0] : period.split('_')[1]]
            
            data_rcp45[data_rcp45==0] = 0.00000000001
            data_rcp85[data_rcp85==0] = 0.00000000001

            data_rcp45_year = data_rcp45.dropna().resample('A').sum()
            data_rcp85_year = data_rcp85.dropna().resample('A').sum()


            Analisis_models_rcp45.loc[nmod,:2] = n_meses_aport(data_rcp45.iloc[:,0].astype(float),np.percentile(SIM_Hist.values,25))
            Analisis_models_rcp85.loc[nmod,:2] = n_meses_aport(data_rcp85.iloc[:,0].astype(float),np.percentile(SIM_Hist.values,25))
            Analisis_models_rcp45.loc[nmod,3] = SPI_CC(SIM_Hist.iloc[:,0].astype(float),data_rcp45.iloc[:,0].dropna().astype(float), verbose=False).mean()
            Analisis_models_rcp85.loc[nmod,3] = SPI_CC(SIM_Hist.iloc[:,0].astype(float),data_rcp85.iloc[:,0].dropna().astype(float), verbose=False).mean()

        Analisis_mensual_CC[period]['RCP_45']['Nº de meses con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_rcp45.mean()[0])
        Analisis_mensual_CC[period]['RCP_45']['Nº de meses con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_rcp45.max()[0]
        Analisis_mensual_CC[period]['RCP_45']['Nº de meses con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_rcp45.min()[0]

        Analisis_mensual_CC[period]['RCP_45']['Nº de meses consecutivos con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_rcp45.mean()[1])
        Analisis_mensual_CC[period]['RCP_45']['Nº de meses consecutivos con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_rcp45.max()[1]
        Analisis_mensual_CC[period]['RCP_45']['Nº de meses consecutivos con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_rcp45.min()[1]

        Analisis_mensual_CC[period]['RCP_45']['Nº meses con aportaciones == 0']['Mean'].loc[name_embalse] = int(Analisis_models_rcp45.mean()[2])
        Analisis_mensual_CC[period]['RCP_45']['Nº meses con aportaciones == 0']['Max'].loc[name_embalse]  = Analisis_models_rcp45.max()[2]
        Analisis_mensual_CC[period]['RCP_45']['Nº meses con aportaciones == 0']['Min'].loc[name_embalse]  = Analisis_models_rcp45.min()[2]

        Analisis_mensual_CC[period]['RCP_45']['SSFI']['Mean'].loc[name_embalse] = Analisis_models_rcp45.mean()[3]
        Analisis_mensual_CC[period]['RCP_45']['SSFI']['Max'].loc[name_embalse]  = Analisis_models_rcp45.max()[3]
        Analisis_mensual_CC[period]['RCP_45']['SSFI']['Min'].loc[name_embalse]  = Analisis_models_rcp45.min()[3]



        Analisis_mensual_CC[period]['RCP_85']['Nº de meses con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_rcp85.mean()[0])
        Analisis_mensual_CC[period]['RCP_85']['Nº de meses con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_rcp85.max()[0]
        Analisis_mensual_CC[period]['RCP_85']['Nº de meses con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_rcp85.min()[0]

        Analisis_mensual_CC[period]['RCP_85']['Nº de meses consecutivos con aportaciones < Q25']['Mean'].loc[name_embalse] = int(Analisis_models_rcp85.mean()[1])
        Analisis_mensual_CC[period]['RCP_85']['Nº de meses consecutivos con aportaciones < Q25']['Max'].loc[name_embalse]  = Analisis_models_rcp85.max()[1]
        Analisis_mensual_CC[period]['RCP_85']['Nº de meses consecutivos con aportaciones < Q25']['Min'].loc[name_embalse]  = Analisis_models_rcp85.min()[1]

        Analisis_mensual_CC[period]['RCP_85']['Nº meses con aportaciones == 0']['Mean'].loc[name_embalse] = int(Analisis_models_rcp85.mean()[2])
        Analisis_mensual_CC[period]['RCP_85']['Nº meses con aportaciones == 0']['Max'].loc[name_embalse]  = Analisis_models_rcp85.max()[2]
        Analisis_mensual_CC[period]['RCP_85']['Nº meses con aportaciones == 0']['Min'].loc[name_embalse]  = Analisis_models_rcp85.min()[2]

        Analisis_mensual_CC[period]['RCP_85']['SSFI']['Mean'].loc[name_embalse] = Analisis_models_rcp85.mean()[3]
        Analisis_mensual_CC[period]['RCP_85']['SSFI']['Max'].loc[name_embalse]  = Analisis_models_rcp85.max()[3]
        Analisis_mensual_CC[period]['RCP_85']['SSFI']['Min'].loc[name_embalse]  = Analisis_models_rcp85.min()[3]

    Analisis_mensual_hist.to_excel(path_project+'/06_ANALISIS_RESULTADOS/'+'Indicadores_Hist.xlsx')
    Analisis_mensual_CC.to_excel(path_project+'/06_ANALISIS_RESULTADOS/'+'Indicadores_CC.xlsx')
    
    
def figuras_incertidumbre(Hist,RCP45,RCP85,var):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if var=='Prec':
        Hist_year  = Hist.dropna().resample('A').sum()
        RCP45_year = RCP45.dropna().resample('A').sum()
        RCP85_year = RCP85.dropna().resample('A').sum()
    else:
        Hist_year  = Hist.dropna().resample('A').mean()
        RCP45_year = RCP45.dropna().resample('A').mean()
        RCP85_year = RCP85.dropna().resample('A').mean()
        
    
    
    lim_max = np.max([Hist_year.max(),RCP45_year.max(),RCP85_year.max()])
    lim_min = np.min([Hist_year.min(),RCP45_year.min(),RCP85_year.min()])
    
    
    Hist_year.rolling(3).mean().mean(axis=1).plot(kind='line',color='darkgray',ax=ax,label='Hist')
    RCP45_year.rolling(3).mean().mean(axis=1).plot(kind='line',color='blue',ax=ax,label='RCP 4.5')
    RCP85_year.rolling(3).mean().mean(axis=1).plot(kind='line',color='red',ax=ax,label='RCP 8.5')


    ax.vlines('2006-1-31', lim_max, lim_min, 'k', linestyle = '-', linewidth = 2)

    Hist_year.quantile(0.25,axis=1).plot(linestyle = '--', color='darkgray', alpha=0.5) 
    Hist_year.quantile(0.75,axis=1).plot(linestyle = '--', color='darkgray', alpha=0.5)
    ax.fill_between(Hist_year.index, Hist_year.quantile(0.25,axis=1),Hist_year.quantile(0.75,axis=1), color='darkgray', alpha=0.1)

    RCP45_year.quantile(0.25,axis=1).plot(linestyle = '--', color='blue',  alpha=0.5) 
    RCP45_year.quantile(0.75,axis=1).plot(linestyle = '--', color='blue',  alpha=0.5)
    ax.fill_between(RCP45_year.index, RCP45_year.quantile(0.25,axis=1),RCP45_year.quantile(0.75,axis=1), color='blue', alpha=0.1)


    RCP85_year.quantile(0.25,axis=1).plot(linestyle = '--', color='red',  alpha=0.5) 
    RCP85_year.quantile(0.75,axis=1).plot(linestyle = '--', color='red',  alpha=0.5)
    ax.fill_between(RCP85_year.index, RCP85_year.quantile(0.25,axis=1),RCP85_year.quantile(0.75,axis=1), color='red', alpha=0.1)

    ax.set_ylim(lim_min,lim_max)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines[:3],labels[:3],loc=2)
    if var=='Prec':
        ax.set_ylabel('Precipitación (mm/año)')
    else:
        ax.set_ylabel('Temperatura (ºC)')
        
    # ax.set_title('Cambio en la precipitación media de '+cuenca , y = 1)
    
    

    
        
    
