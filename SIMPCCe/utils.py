'''
La librería contiene funciones complementarias a la metodología.
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pingouin as pg
from scipy.stats import multivariate_normal
from matplotlib import cm
from scipy import integrate
from numpy import trapz
from matplotlib.patches import Polygon
#import pyeto

def create_name(climate_change,var,rcp,m):
    if climate_change=='CORDEX':
        name_file_fut = f'{var}_month_CORDEX_{rcp}_{m[0]}_r1i1p1'
    else:
        name_file_fut = f'{var}_month_{m[0]}_{rcp}_{m[1]}_{m[2]}'
    return name_file_fut



def text(x, y, text,angle,fontsize,ax):
    ax.text(x, y, text,
            ha='center', va='top', weight='bold',backgroundcolor = 'white',rotation=angle,fontsize=fontsize)

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


def extract_aportaciones_QQM(path_project,rcp, period):
    
    models = ['CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5',
       'CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES',
       'CLMcom-CCLM4-8-17_MPI-M-MPI-ESM-LR',
       'KNMI-RACMO22E_ICHEC-EC-EARTH', 'KNMI-RACMO22E_MOHC-HadGEM2-ES',
       'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR',
       'SMHI-RCA4_CNRM-CERFACS-CNRM-CM5', 'SMHI-RCA4_IPSL-IPSL-CM5A-MR',
       'SMHI-RCA4_MOHC-HadGEM2-ES', 'SMHI-RCA4_MPI-M-MPI-ESM-LR']
    
    path = path_project +'05_CAMBIO_CLIMATICO/02_APORTACIONES/'
    
    year_ini = period[:4]
    year_fin = period[-4:]
    
    data = pd.DataFrame(index=pd.date_range(start=year_ini+'-01-01',end=year_fin+'-12-31',freq='M'),columns=models)
    
    for i in models:
        data_model = pd.read_csv(path+'Aportaciones.'+i+'.'+rcp+'.csv',index_col=0, parse_dates=True)
        year_fin = str(np.min([int(year_fin),np.max(data_model.index.year)]))
        
        data_model = data_model.loc[year_ini:year_fin]
        
        data.loc[year_ini:year_fin,i] = data_model.values.flatten()
        
    return data


def test_ANOVA(path_project):
    data_RCP45 = extract_aportaciones_QQM(path_project,'RCP_45', '2011_2098')
    data_RCP85 = extract_aportaciones_QQM(path_project,'RCP_85', '2011_2098')

    ANOVA_DF = pd.DataFrame(index=data_RCP45.index,columns=['RCP','MODELS','RCP+MODELS'])

    df_contruc = pd.DataFrame(index=np.arange(0,len(data_RCP45)*2*10),columns=['Time','Aportaciones','Model','RCP'])

    df_contruc.loc[:,'Time']         = np.tile((np.repeat(data_RCP45.index,10)),2)
    df_contruc.loc[:,'Aportaciones'] = pd.concat((data_RCP45,data_RCP85)).values.flatten().astype(float).reshape(-1,1)
    df_contruc.loc[:,'Model']        = np.tile((data_RCP45.columns),len(data_RCP45)*2)
    df_contruc.loc[:,'RCP']          = np.repeat(['RCP_45','RCP_85'],len(data_RCP45)*10)
    df_contruc = df_contruc.dropna()

    for t,i in enumerate(df_contruc.groupby('Time')):
        data=i[1]
        ANOVA_DF.iloc[t,:]  = pg.anova(
            data     = data,
            dv       = 'Aportaciones',
            between  = ['RCP','Model'],
            detailed = True
        ).SS.values[:-1]


    ANOVA_DF_Y = ANOVA_DF.resample('A').sum()
    normalized_df=ANOVA_DF_Y/ANOVA_DF_Y.max()
    
    
    normalized_df.iloc[:,0].rolling(window =10).mean().plot(label = 'RCP')
    normalized_df.iloc[:,1].rolling(window =10).mean().plot(label = 'MODELOS')
    normalized_df.iloc[:,2].rolling(window =10).mean().plot(label = 'RCP+MODELOS')
    plt.ylabel('Incertidumbre')
    plt.legend()
    
    
def perturbate_serie(serie_hist,serie_hist_mod,serie_fut):
    serie_hist[serie_hist==0]= 0.0000001
    serie_hist_mod[serie_hist_mod==0]= 0.0000001
    serie_fut[serie_fut==0]= 0.0000001
    
    
    
    xc   = serie_hist.resample('Y').mean().values
    xc_m = serie_hist_mod.resample('Y').mean().values
    xf_m = serie_fut.resample('Y').mean().values
    
    var_mean = (xf_m.mean()-xc_m.mean())/xc_m.mean()
    var_CV   = (np.std(xf_m)/np.mean(xf_m)-np.std(xc_m)/np.mean(xc_m))/(np.std(xc_m)/np.mean(xc_m))
    
    x1 = xc/xc.mean()
    x2 = ((x1-1)*(1+var_CV))+1
    x3 = x2*xc.mean()*(1+var_mean)
    
    serie_CC_real = pd.DataFrame(index=serie_fut.index,columns=['H3/mes'])
    
    Rm = serie_fut.groupby(by = serie_fut.index.month).mean().values/serie_hist_mod.groupby(by = serie_hist_mod.index.month).mean().values
    dm = serie_hist.groupby(by = serie_hist.index.month).mean().values/np.sum(serie_hist.groupby(by = serie_hist.index.month).mean().values)

    y = 0
    for i in range(0,len(serie_fut),12):
        serie_CC_real.iloc[i:i+12,0] = (dm * Rm * 1/(np.sum(Rm * dm)) * x3[y]*12).flatten()
        y=y+1
    
    
    return serie_CC_real


def dam_gestion_model(Aportaciones,volumen_embalse,demanda_anual,coef_mensual_demanda,volumen_inicial,volumen_emergencia):
    df_suministro            = pd.DataFrame(index=Aportaciones.index, columns=['Hm3'])
    df_volumen_almacenado    = pd.DataFrame(index=Aportaciones.index, columns=['Hm3'])
    df_volumen_aliviado      = pd.DataFrame(index=Aportaciones.index, columns=['Hm3'])

    serie_demanda = np.tile(coef_mensual_demanda,len(np.unique(Aportaciones.index.year)))*demanda_anual
    volumen_inst  = volumen_inicial
    for i in range(len(Aportaciones.index)):
        demanda_inst = serie_demanda[i]

        volumen_inst = volumen_inst+Aportaciones.iloc[i].values
        volumen_aliviado = volumen_inst-volumen_embalse

        if volumen_aliviado<0:
            volumen_aliviado = 0

        if volumen_inst>volumen_embalse:
            volumen_inst = volumen_embalse

        if volumen_inst<=volumen_emergencia:
            volumen_satis = 0
            volumen_inst  = volumen_inst-volumen_satis
        
        elif volumen_inst>demanda_inst:
            volumen_satis = demanda_inst
            volumen_inst  = volumen_inst-volumen_satis

        elif (volumen_inst <demanda_inst):
            volumen_satis = volumen_inst
            #volumen_inst  = volumen_inst-volumen_satis
            
            
#         if volumen_inst<0:
#             volumen_inst = 0
#         if volumen_satis<0:
#             volumen_satis = 0
        


        df_suministro.iloc[i,:]         = volumen_satis
        df_volumen_almacenado.iloc[i,:] = volumen_inst
        df_volumen_aliviado.iloc[i,:]   = volumen_aliviado
        
    return df_suministro, df_volumen_almacenado, df_volumen_aliviado,serie_demanda


def demand_realability_curve(demanda_anual,serie_sumunistro_mensual,plot=True):
    sort = np.sort(serie_sumunistro_mensual.resample('A').sum().values.flatten())
    exceedence = np.arange(1.,len(sort)+1) / len(sort)
    exceedence_interp = np.interp(np.arange(0,1,0.0005), exceedence, sort)
    
    pos_rk = np.where(np.arange(0,1,0.0005)==0.85)[0][0]
    curve_sk = sorted(exceedence_interp,reverse=True)
    curve_sk =curve_sk[:pos_rk]
    
    I1k = np.max(sort)/demanda_anual
    I2k = trapz(curve_sk, dx=5)/trapz(demanda_anual*np.ones(len(curve_sk)), dx=5)
    
    if plot==True:

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0,100)
        ax.set_ylim(np.min(sort),demanda_anual+30)
        ax.plot(sorted(np.arange(0,1,0.0005)*100,reverse=True), exceedence_interp, linestyle='-', color='darkblue', label = 'Simulado')
        ax.hlines(y=demanda_anual,xmin=0,xmax=100,linestyle='--', color='k')

        ax.set_yticks([sorted(exceedence_interp,reverse=True)[pos_rk],np.max(sorted(exceedence_interp)),demanda_anual])
        ax.set_yticklabels(['Suministro\n Aceptable','Suministro','Demanda'],weight='bold')
        ax.hlines(sorted(exceedence_interp,reverse=True)[pos_rk],xmin=0,xmax=85,linestyle='--',color='red')
        ax.vlines(85,ymin=0,ymax=sorted(exceedence_interp,reverse=True)[pos_rk],linestyle='--',color='red')
        ax.hlines(np.max(sorted(exceedence_interp)),xmin=0,xmax=100,linestyle='--',color='red')

        ax2 = ax.twinx()
        ax2.set_yticks([sorted(exceedence_interp,reverse=True)[pos_rk],np.max(sorted(exceedence_interp)),demanda_anual])
        ax2.set_ylim(np.min(sort),demanda_anual+30)
        ax2.set_ylabel('Volumen '+r'$H^{3}$'+'/Año',fontsize=15)
        ax.set_xlabel('Fiabilidad(%)',fontsize=15)

        text(85, (np.min(sorted(exceedence_interp-10))), 'R=85%',0,12,ax)
        ax.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
    
    
    
    return I1k,I2k

def diagnosis_severidad(I1,I2):
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

    fig = plt.figure()
    ax1 = fig.add_axes((0.1,0.3,0.8,0.8)) # create an Axes with some room below

    X =  np.linspace(0,1,1000)
    Y =  np.linspace(0,1,1000)

    ax1.vlines(0.70,0,1)
    ax1.vlines(0.85,0,1)

    ax1.hlines(0.60,0,1)
    ax1.hlines(0.75,0,1)

    ax1.plot(X,Y)
    ax1.plot(I1,I2,'bo', markersize=12)
    #ax1.set_xticks(np.arange(0,1.2,0.2))
    #ax1.set_xticklabels(np.arange(0,1.2,0.2).round(2).astype(str))
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    #ax1.fill_between(x1,y1, color="none", hatch="X", edgecolor="b", linewidth=0.0)
    ax1.add_patch(Polygon([(0,0), (0.7,0),(0.7,0.6),(0.6, 0.6)],
                           closed=True, facecolor='red',alpha=0.4))
    ax1.add_patch(Polygon([(0.7,0.6), (0.7,0.7),(0.6,0.6)],
                           closed=True, facecolor='grey',alpha=0.6))
    ax1.add_patch(Polygon([(0.7,0), (1,0),(1,0.6),(0.7,0.6)],
                           closed=True, facecolor='grey',alpha=0.6))
    ax1.add_patch(Polygon([(0.7,0.6), (1,0.6),(1,0.75),(0.7,0.75)],
                           closed=True, facecolor='grey',alpha=0.2))
    ax1.add_patch(Polygon([(0.75,0.75), (1,0.75),(1,1)],
                           closed=True, facecolor='blue',alpha=0.2))
    
    red_patch   = mpatches.Patch(color='red',alpha=0.4,label = 'Problema muy serio')
    grey1_patch = mpatches.Patch(color='grey',alpha=0.6,label= 'Problema serio')
    grey2_patch = mpatches.Patch(color='grey',alpha=0.2,label= 'Problema medio')
    blue_patch  = mpatches.Patch(color='blue',alpha=0.2,label= 'Sin problema')



    # create second Axes. Note the 0.0 height
    ax2 = fig.add_axes((0.1,0.15,0.8,0.0))
    ax2.yaxis.set_visible(False) # hide the yaxis


    ax3 = fig.add_axes([-0.02, 0.3, 0, 0.8])
    ax3.yaxis.set_visible(True) # hide the yaxis
    ax3.xaxis.set_visible(False) # hide the yaxis

    new_tick_locations = np.array([0,0.7, 0.85, 1])

    def tick_function(X):
        V = 1/(1+X)
        return ["%.3f" % z for z in V]

    # ax2.set_xticks(new_tick_locations)
    # ax2.set_xticklabels([0,0.7, 0.85, 1])


    ax2.set_xticks([0,0.7,0.85,1])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.set_yticks([0,0.6,0.75,1])
    plt.setp(ax3.get_yticklabels(), visible=False)

    text(0.35,-0.2, "No favorable",0,10,ax1)
    text(0.775,-0.2, "Neutral",0,10,ax1)
    text(0.95,-0.2, "Favorable",0,10,ax1)

    text(-0.2,0.475, "No favorable",90,10,ax1)
    text(-0.2,0.77, "Neutral",90,10,ax1)
    text(-0.2,1.05, "Favorable",90,10,ax1)
    #ax2.set_xticklabels([0,0.7,0.85,1])

    ax1.set_xlabel(r'$I_{1}$')
    ax1.set_ylabel(r'$I_{2}$')
    ax1.grid()
    
    fig.legend(handles=[red_patch,grey1_patch,grey2_patch,blue_patch],ncol=1,fontsize=12,  bbox_to_anchor=[1.25, 0.8])
    
def calculate_volumen_scipy(rv):
    def return_pdf(y,x,fit=rv):
        pdf = rv.pdf([x,y])
        return pdf
    x, y = np.linspace(0, 1, 1000), np.linspace(0, 1, 1000)
    x_z, y_z = np.linspace(0, 10, 1000), np.linspace(0, 10, 1000)
    xx,yy    = np.meshgrid(x,y)
    z        = np.zeros(xx.shape)
    
    z[(xx>=yy)&(xx<=0.7)&(yy<=0.6)] = 1 #zona1
    z[(xx>=yy)&(xx<=0.7)&(yy>0.6)]  = 2 #zona2
    z[(xx>0.7)&(yy<=0.6)]           = 2 #zona2
    z[(xx>0.7)&(yy>0.6)&(yy<=0.75)] = 3 #zona3
    z[(xx>=yy)&(yy>0.75)]           = 4 #zona4

    z1 = np.zeros(xx.shape); z1[z==1] = 1
    z2 = np.zeros(xx.shape); z1[z==2] = 1
    z3 = np.zeros(xx.shape); z1[z==3] = 1
    z4 = np.zeros(xx.shape); z1[z==4] = 1

    pos1  = np.dstack((xx*z1, yy*z1))
    pos2  = np.dstack((xx*z2, yy*z2))
    pos3  = np.dstack((xx*z3, yy*z3))
    pos4  = np.dstack((xx*z4, yy*z4))

    result1   = integrate.dblquad(return_pdf, 0,0.6, lambda x: 0, lambda x: x)[0]
    result1   = result1+integrate.dblquad(return_pdf, 0.6,0.7, lambda x: 0, lambda x: 0.6)[0]
    
    result2   = integrate.dblquad(return_pdf, 0.7,1, lambda x: 0, lambda x: 0.6)[0]
    result2   = result2+integrate.dblquad(return_pdf, 0.6,0.7, lambda x: 0.6, lambda x: x)[0]
    
    result3   = integrate.dblquad(return_pdf, 0.7,1, lambda x: 0.6, lambda x: 0.75)[0]
    result4   = integrate.dblquad(return_pdf, 0.7,1, lambda x: 0.75, lambda x: x)[0]
    
    result = result1+result2+result3+result4
    
    result_z = integrate.dblquad(return_pdf, 1,100, lambda x: 0, lambda x: 1)[0]
    result_final = result1/result+result2/result+result3/result+result4/result
    
    result_z = integrate.dblquad(return_pdf, 1,100, lambda x: 0, lambda x: 1)[0]
    # result_z = result_z+integrate.dblquad(return_pdf, 0,1, lambda x: 1, lambda x: 1)[0]-integrate.dblquad(return_pdf, 0,1, lambda x: 1, lambda x: x)[0]
    # result_z = result_z-integrate.dblquad(return_pdf, 0.7,0.75, lambda x: 0.7, lambda x: x)[0]

    #result_final = result+result_z
    print(result)
    
    return result_final, result1/result, result2/result, result3/result,result4/result


def diagnosis_severidad_CC(I1_hist,I2_hist,I1_rcp45,I2_rcp45,I1_rcp85,I2_rcp85,title):
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

    fig = plt.figure()
    ax1 = fig.add_axes((0.1,0.3,0.8,0.8)) # create an Axes with some room below

    X =  np.linspace(0,1,1000)
    Y =  np.linspace(0,1,1000)

    ax1.vlines(0.70,0,1)
    ax1.vlines(0.85,0,1)

    ax1.hlines(0.60,0,1)
    ax1.hlines(0.75,0,1)

    ax1.plot(X,Y)
    
    #ax1.set_xticks(np.arange(0,1.2,0.2))
    #ax1.set_xticklabels(np.arange(0,1.2,0.2).round(2).astype(str))
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    #ax1.fill_between(x1,y1, color="none", hatch="X", edgecolor="b", linewidth=0.0)
    ax1.add_patch(Polygon([(0,0), (0.7,0),(0.7,0.6),(0.6, 0.6)],
                           closed=True, facecolor='red',alpha=0.4))
    ax1.add_patch(Polygon([(0.7,0.6), (0.7,0.7),(0.6,0.6)],
                           closed=True, facecolor='grey',alpha=0.6))
    ax1.add_patch(Polygon([(0.7,0), (1,0),(1,0.6),(0.7,0.6)],
                           closed=True, facecolor='grey',alpha=0.6))
    ax1.add_patch(Polygon([(0.7,0.6), (1,0.6),(1,0.75),(0.7,0.75)],
                           closed=True, facecolor='grey',alpha=0.2))
    ax1.add_patch(Polygon([(0.75,0.75), (1,0.75),(1,1)],
                           closed=True, facecolor='blue',alpha=0.2))
    
    red_patch   = mpatches.Patch(color='red',alpha=0.4,label = 'Problema muy serio')
    grey1_patch = mpatches.Patch(color='grey',alpha=0.6,label= 'Problema serio')
    grey2_patch = mpatches.Patch(color='grey',alpha=0.2,label= 'Problema medio')
    blue_patch  = mpatches.Patch(color='blue',alpha=0.2,label= 'Sin problema')



    # create second Axes. Note the 0.0 height
    ax2 = fig.add_axes((0.1,0.15,0.8,0.0))
    ax2.yaxis.set_visible(False) # hide the yaxis


    ax3 = fig.add_axes([-0.02, 0.3, 0, 0.8])
    ax3.yaxis.set_visible(True) # hide the yaxis
    ax3.xaxis.set_visible(False) # hide the yaxis

    new_tick_locations = np.array([0,0.7, 0.85, 1])

    def tick_function(X):
        V = 1/(1+X)
        return ["%.3f" % z for z in V]

    # ax2.set_xticks(new_tick_locations)
    # ax2.set_xticklabels([0,0.7, 0.85, 1])


    ax2.set_xticks([0,0.7,0.85,1])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.set_yticks([0,0.6,0.75,1])
    plt.setp(ax3.get_yticklabels(), visible=False)

    text(0.35,-0.2, "No favorable",0,10,ax1)
    text(0.775,-0.2, "Neutral",0,10,ax1)
    text(0.95,-0.2, "Favorable",0,10,ax1)

    text(-0.2,0.475, "No favorable",90,10,ax1)
    text(-0.2,0.77, "Neutral",90,10,ax1)
    text(-0.2,1.05, "Favorable",90,10,ax1)
    #ax2.set_xticklabels([0,0.7,0.85,1])
    
    x_cc = np.concatenate((I1_rcp45,I1_rcp85))
    y_cc = np.concatenate((I2_rcp45,I2_rcp85))

    xx, yy = np.mgrid[0:1:.01, 0:1:.01]
    pos = np.dstack((xx, yy))
    step = 5

    mean = np.mean([x_cc,y_cc], axis=1)
    cov = np.cov([x_cc,y_cc], rowvar=1)
    
    
    rv = multivariate_normal(mean,cov)

    Z =  rv.pdf(pos)

    m = np.amax(Z)

    levels = np.arange(0.0, m+10, step) + step

    cs = ax1.contour(xx, yy, Z,cmap=cm.seismic,levels=10)
    
    hist = ax1.plot(I1_hist,I2_hist,'ko', markersize=8,label='Hist')
    rcp45 = ax1.plot(I1_rcp45,I2_rcp45,'bo', markersize=3,label='RCP 4.5')
    rcp85 = ax1.plot(I1_rcp85,I2_rcp85,'ro', markersize=3,label='RCP 8.5')

    ax1.set_xlabel(r'$I_{1}$')
    ax1.set_ylabel(r'$I_{2}$')
    ax1.grid()
    
    [Zona_total,Zona_1,Zona_2,Zona_3,Zona_4] = np.abs(calculate_volumen_scipy(rv))
    
    ax4 = fig.add_axes((0.2,-0.4,0.8,0.8))
    ax4.axis('off')
    
    Tabla = pd.DataFrame(columns=["Problema muy serio","Problema Serio","Problema medio","Sin Problema"],index=['Probabilidad (%)'])
    Tabla.iloc[0,:] = [Zona_1,Zona_2,Zona_3,Zona_4]
    print(Zona_total)
    Tabla = Tabla.astype(float)*100
    Tabla = Tabla.round(2)
    print(Tabla)
    the_table = ax4.table(cellText=Tabla.values,
                                rowLabels=Tabla.index,
                                colLabels=["Problema muy serio","Problema Serio","Problema medio","Sin Problema"],
                                loc="center")

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 2.3)
       
    
    fig.legend(handles=[red_patch,grey1_patch,grey2_patch,blue_patch,hist[0],rcp45[0],rcp85[0]],ncol=1,fontsize=12,  bbox_to_anchor=[1.3, 0.95])


def perturbate_serie_with_coeffs(serie_hist, serie_hist_mod, serie_fut, var_mean, var_CV):
    # Asegura no ceros
    serie_hist[serie_hist==0] = 0.0000001
    serie_hist_mod[serie_hist_mod==0] = 0.0000001
    serie_fut[serie_fut==0] = 0.0000001

    # Cálculos base
    xc = serie_hist.resample('Y').mean().values
    x1 = xc / xc.mean()
    x2 = ((x1-1)*(1+var_CV)) + 1
    x3 = x2 * xc.mean() * (1+var_mean)

    # Inicializa serie perturbada
    serie_CC_real = pd.DataFrame(index=serie_fut.index, columns=['H3/mes'])

    # Calcula Rm y dm mensuales
    Rm = serie_fut.groupby(by = serie_fut.index.month).mean().values / serie_hist_mod.groupby(by = serie_hist_mod.index.month).mean().values
    dm = serie_hist.groupby(by = serie_hist.index.month).mean().values / np.sum(serie_hist.groupby(by = serie_hist.index.month).mean().values)

    # Aplica la perturbación mes a mes
    y = 0
    for i in range(0,len(serie_fut),12):
        serie_CC_real.iloc[i:i+12,0] = (dm * Rm * 1/(np.sum(Rm * dm)) * x3[y]*12).flatten()
        y += 1

    return serie_CC_real

def perturbate_and_save_all_models(path_project, serie_hist_user, climate_change, models, periodos_fut):
    esce = ['ssp245','ssp585'] if climate_change == 'CMIP6' else ['rcp45','rcp85']

    # Asegura no ceros en histórico
    serie_hist = serie_hist_user.loc['1995':'2014']
    serie_hist[serie_hist==0]= 0.0000001

    for model in models:
        print(f"\n🔹 Procesando modelo: {model}")
        modelo_split = model.split("_")

        # Lee serie histórica modelada
        file_hist_mod = f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones','historical',modelo_split)}.csv"
        serie_hist_mod = pd.read_csv(file_hist_mod, index_col=0, parse_dates=True)
        serie_hist_mod[serie_hist_mod==0]= 0.0000001

        for scenario in esce:
            print(f"   ➔ Escenario: {scenario}")
            # Inicializa lista para concatenar periodos futuros perturbados de este escenario
            series_futuras = []

            for period in periodos_fut:
                # Lee serie futura de este modelo, escenario y periodo
                file_fut = f"{path_project}/05_CAMBIO_CLIMATICO/02_APORTACIONES/{create_name(climate_change,'Aportaciones',scenario,modelo_split)}.csv"
                serie_fut = pd.read_csv(file_fut, index_col=0, parse_dates=True)
                serie_fut = serie_fut.loc[period.split('_')[0]:period.split('_')[1]]
                serie_fut[serie_fut<0]= 0.0000001

                # Calcula coeficientes para este modelo, escenario y periodo
                xc_m = serie_hist_mod.resample('Y').mean().values
                xf_m = serie_fut.resample('Y').mean().values

                var_mean = (xf_m.mean()-xc_m.mean())/xc_m.mean()
                var_CV   = (np.std(xf_m)/np.mean(xf_m) - np.std(xc_m)/np.mean(xc_m)) / (np.std(xc_m)/np.mean(xc_m))

                # Perturba
                serie_CC_real = perturbate_serie_with_coeffs(serie_hist.copy(), serie_hist_mod, serie_fut, var_mean, var_CV)
                serie_CC_real[serie_CC_real<0] = 0
                serie_CC_real.columns = serie_hist_user.columns

                series_futuras.append(serie_CC_real)

            # Concatena histórico y futuros para este escenario
            serie_total = pd.concat([serie_hist_user.loc[:'2020']] + series_futuras, axis=0)

            # Guarda la serie perturbada completa del modelo y escenario
            output_dir = os.path.join(path_project, '06_ANALISIS_RESULTADOS', 'Series_Perturbadas_Modelos')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'Serie_Perturbada_{model}_{scenario}.csv')
            serie_total.to_csv(output_file)

            print(f"   ✅ Serie perturbada guardada en: {output_file}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np

def process_and_perturb_historical_series(
    file_aportaciones,
    embalse,
    path_project,
    models,
    climate_change,
    periodos_fut = ['2021_2040','2041_2060','2061_2080','2081_2100']
):
    """
    Lee la serie histórica de aportaciones, la procesa y ejecuta la perturbación climática
    para los periodos futuros definidos, guardando los resultados.

    Parameters
    ----------
    file_aportaciones : str
        Ruta al archivo Excel de aportaciones históricas.

    embalse : str
        Nombre del embalse a filtrar en el DataFrame.

    path_project : str
        Ruta base del proyecto donde se guardarán los resultados.

    climate_change : str
        dataset u objeto de cambio climático.
    
    models : list
        lista de modelos climáticos.

    periodos_fut : list of str, optional
        Lista de periodos futuros de simulación. Default: ['2021_2040','2041_2060','2061_2080','2081_2100'].

    Returns
    -------
    None
        Procesa y guarda las series perturbadas para todos los modelos y periodos definidos.
    """

    # ➡️ Lee la serie histórica desde Excel
    serie_hist_ = pd.read_excel(
        file_aportaciones,
        index_col=0, parse_dates=True, skiprows=1
    )

    # Reemplaza valores 'No data' por NaN
    serie_hist_.replace('No data', np.nan, inplace=True)

    # Convierte todas las columnas a valores numéricos
    serie_hist_num = serie_hist_.apply(pd.to_numeric, errors='coerce')

    # Re-muestrea la serie a escala mensual sumando los valores diarios
    serie_hist_monthly = serie_hist_num.resample('M').sum()

    # Filtra y limpia la serie del embalse específico
    if embalse not in serie_hist_monthly.columns:
        print(f"⚠️ Embalse {embalse} no encontrado en el archivo de aportaciones.")
        return

    serie_hist = pd.DataFrame(serie_hist_monthly.loc[:, embalse])
    serie_hist[serie_hist < 0] = 0

    # ➡️ Ejecuta la función de perturbación y guardado
    perturbate_and_save_all_models(path_project, serie_hist, climate_change, models, periodos_fut)

    print(f"✅ Series perturbadas generadas y guardadas para {embalse}.")


import os
import pandas as pd
import numpy as np

def process_ensemble_for_embalse(
    nombre_embalse,
    file_aportaciones,
    path_project,
    models,
    escenarios = ['ssp245', 'ssp585']
):
    """
    Procesa series históricas y calcula estadísticos ensemble para un embalse y múltiples escenarios climáticos.

    Parameters
    ----------
    nombre_embalse : str
        Nombre del embalse a procesar.

    file_aportaciones : str
        Ruta al archivo Excel de aportaciones históricas.

    path_project : str
        Ruta base del proyecto del embalse.

    models : list of str
        Lista de modelos climáticos a procesar.

    climate_change : object
        Objeto de cambio climático (no usado directamente aquí pero se incluye para consistencia con otros flujos).

    escenarios : list of str, optional
        Escenarios climáticos a procesar. Default: ['ssp245', 'ssp585'].

    Returns
    -------
    None
        Calcula y guarda los archivos de estadísticos ensemble en la carpeta del embalse.
    """

    print(f"Procesando embalse: {nombre_embalse}")

    # =========== 1. Leer serie histórica ===========
    serie_hist_ = pd.read_excel(
        file_aportaciones,
        index_col=0, parse_dates=True, skiprows=1
    )
    serie_hist_.replace('No data', np.nan, inplace=True)
    serie_hist_num = serie_hist_.apply(pd.to_numeric, errors='coerce')
    serie_hist_monthly = serie_hist_num.resample('M').sum()

    if nombre_embalse not in serie_hist_monthly.columns:
        print(f"⚠️ Embalse {nombre_embalse} no encontrado en el archivo de aportaciones.")
        return

    serie_hist = pd.DataFrame(serie_hist_monthly.loc[:, nombre_embalse])
    serie_hist[serie_hist < 0] = 0

    # =========== 2. Itera escenarios ssp245 y ssp585 ===========
    for esce in escenarios:
        print(f"  Procesando escenario ensemble real: {esce}")

        series_models = []

        for model in models:
            file_perturbed = os.path.join(
                path_project, '06_ANALISIS_RESULTADOS', 'Series_Perturbadas_Modelos',
                f'Serie_Perturbada_{model}_{esce}.csv'
            )
            if os.path.exists(file_perturbed):
                df_model = pd.read_csv(file_perturbed, index_col=0, parse_dates=True)
                df_model.columns = [model]
                series_models.append(df_model)
            else:
                print(f"⚠️ Serie perturbada no encontrada para {model} {esce}")

        # =========== 3. Concatena y calcula estadísticos ensemble ===========
        if len(series_models) > 0:
            df_concat = pd.concat(series_models, axis=1)

            df_stats = pd.DataFrame(index=df_concat.index)
            df_stats['ensemble_median'] = df_concat.median(axis=1)
            df_stats['ensemble_p25'] = df_concat.quantile(0.25, axis=1)
            df_stats['ensemble_p75'] = df_concat.quantile(0.75, axis=1)
            df_stats['ensemble_p95'] = df_concat.quantile(0.95, axis=1)
            df_stats['ensemble_p05'] = df_concat.quantile(0.05, axis=1)

            # =========== 4. Guarda cada serie ensemble ===========
            output_dir = os.path.join(path_project, '06_ANALISIS_RESULTADOS', 'Series_Perturbadas_Ensemble')
            os.makedirs(output_dir, exist_ok=True)

            for stat in df_stats.columns:
                output_file = os.path.join(output_dir, f'Serie_CC_{stat}_{esce}_2021_2100.csv')
                df_stats[[stat]].to_csv(output_file)
                print(f"✅ Serie ({stat}) guardada en: {output_file}")

        else:
            print(f"⚠️ No se encontraron series perturbadas para {esce} en {nombre_embalse}")

def plot_ensemble_subplots_embalse(
    embalse, 
    file_aportaciones,
    path_base_project,
    escenarios = ['ssp245', 'ssp585'],
    quantiles = ['p05','p25', 'median', 'p75', 'p95'],
    period_control = ['1995', '2014'],
    save_fig = False,
    fig_name = 'Ensemble_Subplots_LeyendaDebajo.png'
):
    """
    Genera subplots de series ensemble (ssp245 y ssp585) para un embalse.
    Incluye históricos, medianas, percentiles y medias de referencia.

    Parameters
    ----------
    embalse : str
        Nombre del embalse a procesar.

    file_aportaciones : str
        Ruta al archivo Excel de aportaciones históricas.

    path_base_project : str
        Ruta base donde se encuentra la carpeta del embalse.

    escenarios : list of str, optional
        Lista de escenarios a procesar. Default: ['ssp245', 'ssp585'].

    quantiles : list of str, optional
        Lista de cuantiles/estadísticos a leer. Default: ['p05','p25','median','p75','p95'].

    period_control : list of str, optional
        Periodo de control para calcular la media histórica. Default: ['1995', '2014'].

    save_fig : bool, optional
        Si True, guarda la figura en disco. Default: False.

    fig_name : str, optional
        Nombre del archivo de la figura si save_fig=True. Default: 'Ensemble_Subplots_LeyendaDebajo.png'.

    Returns
    -------
    None
        Muestra (y opcionalmente guarda) la figura procesada.
    """

    path_project = os.path.join(path_base_project)
    output_dir = os.path.join(path_project, '06_ANALISIS_RESULTADOS', 'Series_Perturbadas_Ensemble')

    # ➡️ Lee la serie histórica
    serie_hist_ = pd.read_excel(
        file_aportaciones,
        index_col=0, parse_dates=True, skiprows=1
    )
    serie_hist_.replace('No data', np.nan, inplace=True)
    serie_hist_num = serie_hist_.apply(pd.to_numeric, errors='coerce')
    serie_hist_monthly = serie_hist_num.resample('M').sum()

    # Filtra y limpia la serie del embalse actual
    if embalse not in serie_hist_monthly.columns:
        print(f"⚠️ Embalse {embalse} no encontrado en el archivo de aportaciones.")
        return

    serie_hist = pd.DataFrame(serie_hist_monthly.loc[:, embalse])
    serie_hist[serie_hist < 0] = 0
    serie_hist_annual = serie_hist.resample('A').sum()

    # Calcula la media histórica en periodo de control
    mean_hist = serie_hist_annual.loc[period_control[0]:period_control[1]].mean().values[0]

    # ➡️ Inicializa figura
    fig, axs = plt.subplots(1, len(escenarios), figsize=(7*len(escenarios), 5), sharex=True)
    colors = {'ssp245': 'tab:blue', 'ssp585': 'tab:red'}

    if len(escenarios) == 1:
        axs = [axs]  # Si es solo un escenario, lo convierte a lista

    handles_all = []
    labels_all = []

    # ➡️ Itera sobre escenarios y subplots
    for ax, esce in zip(axs, escenarios):
        series_sce = []
        
        # Lee archivos de cada cuantile
        for quantile in quantiles:
            file_path = os.path.join(output_dir, f'Serie_CC_ensemble_{quantile}_{esce}_2021_2100.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = df.loc['2021':]
                df.columns = [quantile]
                df_annual = df.resample('A').sum()
                series_sce.append(df_annual)
            else:
                print(f"⚠️ Archivo no encontrado: {file_path}")
        
        if len(series_sce) > 0:
            df_concat = pd.concat(series_sce, axis=1)
            
            # Calcula estadísticos
            stats_df = pd.DataFrame(index=df_concat.index)
            for quant in quantiles:
                stats_df[quant] = df_concat[quant]

            # Calcula media anual de la mediana
            median_mean = stats_df['median'].mean()
            
            # Relleno P25-P75
            ax.fill_between(stats_df.index.year, stats_df['p25'], stats_df['p75'],
                            color=colors.get(esce,'tab:grey'), alpha=0.3, label=f'{esce.upper()} P25-P75')
            
            # Relleno P5-P95
            ax.fill_between(stats_df.index.year, stats_df['p05'], stats_df['p95'],
                            color=colors.get(esce,'tab:grey'), alpha=0.1, label=f'{esce.upper()} P5-P95')
            
            # Mediana
            ax.plot(stats_df.index.year, stats_df['median'], color=colors.get(esce,'tab:grey'),
                    linewidth=1.8, label=f'Mediana {esce.upper()}')
            
            # Línea media anual de la mediana
            ax.axhline(median_mean, color=colors.get(esce,'tab:grey'), linestyle='dashed',
                       linewidth=1.2, label=f'Media Mediana {esce.upper()}')
            
            # Serie histórica anual
            ax.plot(serie_hist_annual.loc[:'2020'].index.year, serie_hist_annual.loc[:'2020'].values,
                    color='grey', linewidth=1.2, linestyle='-', label='Histórico')
            
            # Media histórica
            ax.axhline(mean_hist, color='green', linestyle='dashdot',
                       linewidth=1.2, label='Media Histórica')
            
            # Estética
            ax.set_title(f'{esce.upper()} - {embalse}', fontsize=13)
            ax.set_ylabel('Hm³/año', fontsize=11)
            ax.grid(True)
            
            # Recolecta handles y labels
            handles, labels = ax.get_legend_handles_labels()
            handles_all.extend(handles)
            labels_all.extend(labels)

    # ➡️ Elimina duplicados en la leyenda manteniendo orden
    legend_items = dict(zip(labels_all, handles_all))

    # ➡️ Leyenda global debajo de la figura
    fig.legend(legend_items.values(), legend_items.keys(),
               loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.12))

    # ➡️ Etiqueta eje X común
    axs[-1].set_xlabel('Año', fontsize=12)

    # ➡️ Ajusta layout
    plt.tight_layout(rect=[0, 0.01, 1, 1])

    # ➡️ Mostrar figura
    plt.show()

    # ➡️ Guardar figura si se indica
    if save_fig:
        output_fig = os.path.join(path_project, '06_ANALISIS_RESULTADOS', fig_name)
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"✅ Figura guardada en: {output_fig}")
