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