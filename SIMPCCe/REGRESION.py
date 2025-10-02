'''
La librería contiene las clases y funciones para la contrucción, entrenamiento, validación 
y simulación de modelos de regresión.
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pickle
#from sklearnex import patch_sklearn
import hydroeval as he
import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from scipy.interpolate import NearestNDInterpolator


#from dask.distributed import Client
import joblib
#client = Client(processes=False) 
#patch_sklearn()
#from dklearn.grid_search import GridSearchCV as DaskGridSearchCV

def nse_func(x, y):
    """Calcula el NSE entre dos series, asegurando forma 1D"""
    x = np.ravel(x)
    y = np.ravel(y)
    return he.evaluator(he.nse, x, y)[0]

def bias_func(x, y):
    """Calcula el PBIAS entre dos series, asegurando forma 1D"""
    x = np.ravel(x)
    y = np.ravel(y)
    return he.evaluator(he.pbias, x, y)[0]

def rsquared(x, y):
    """Calcula el coeficiente de determinación R² entre dos series"""
    x = np.ravel(x)
    y = np.ravel(y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def plot_val_serie(serie_sim,serie_real,path_output):
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        serie_sim.iloc[:,0].plot(ax=ax,label = 'SIM ANN')
        serie_real.iloc[:,0].plot(ax=ax, xlim=('1995','2005'),label='Obs')
    except:
        serie_sim.plot(ax=ax,label = 'SIM ANN')
        serie_real.plot(ax=ax, xlim=('1995','2005'),label='Obs')

    plt.legend()
    ax.set_ylabel("Aportaciones (Hm3)")
    fig.savefig(path_output+'Validacion_Serie_Temporal.png',bbox_inches='tight')
    
    
def plot_curve_CC(serie_sim,serie_real,path_output):
    fig3, ax3 = plt.subplots(figsize=(8, 8))

    data_sim   = serie_sim.values.astype(float).flatten()
    data_real  = serie_real.values.astype(float).flatten()

    sort_sim = np.sort(data_sim)
    exceedence_sim = np.arange(1.,len(sort_sim)+1) / len(sort_sim)

    sort_real  = np.sort(data_real)
    exceedence = np.arange(1.,len(sort_real)+1) / len(sort_real)

    ax3.plot(sorted(exceedence_sim*100,reverse=True), sort_sim, linestyle='-', color='red', label = 'SIM')
    ax3.plot(sorted(exceedence*100,reverse=True), sort_real,linestyle='-', color='darkblue', label = 'REAL')
    ax3.set_xlabel("Excedencia [%]",fontsize=18)
    ax3.set_ylabel("Aportaciones (Hm3)",fontsize=18)
    ax3.tick_params(axis="x", labelsize=14)
    ax3.tick_params(axis="y", labelsize=14)
    ax3.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend(fontsize=12)
    fig3.tight_layout()
    fig3.savefig(path_output+'Validacion_Curva_Caudales_Clas.png',bbox_inches='tight')

nse_scorer = make_scorer(nse_func, greater_is_better=True)
bias_scorer = make_scorer(bias_func, greater_is_better=False)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, validation_curve, GridSearchCV

def read_series(file_Precipitation, file_Temperatura_Maxima, file_Temperatura_Minima,file_Flow):
    """
    Con esta función se cargan las series y se contruyen las matrices necesarias para contruir los modelos de regresión.
    
    Datos de Entrada:
    -----------------
    file_Precipitation:        string.  Fichero de precipitación
    file_Temperatura_Maxima:   string.  Fichero de temperatura máxima
    file_Temperatura_Mínima:   string.  Fichero de temperatura mínima
    file_Flow              :   string.  Fichero de aportaciones
    
    
    
    Resultados:
    -----------------
    X,Y: array. Matrices con los datos necesarios para el entrenamiento y validación

    """
    Precipitation      = pd.read_csv(file_Precipitation,index_col=0,parse_dates=True)
    Temperatura_Maxima = pd.read_csv(file_Temperatura_Maxima,index_col=0,parse_dates=True)
    Temperatura_Minima = pd.read_csv(file_Temperatura_Minima,index_col=0,parse_dates=True)
    Flow               = pd.read_csv(file_Flow,index_col=0,parse_dates=True)
    
    #Precipitation = pd.DataFrame(Precipitation.sum(axis=1))
    #Temperatura_Maxima = pd.DataFrame(Temperatura_Maxima.mean(axis=1))
    #Temperatura_Minima = pd.DataFrame(Temperatura_Minima.mean(axis=1))
    
    
    
    
    index_time_min  = max([Precipitation.index.min(),Flow.index.min()])
    index_time_max  = min([Precipitation.index.max(),Flow.index.max()])
    
    Precipitation = Precipitation.loc[index_time_min:index_time_max]
    Temperatura_Maxima = Temperatura_Maxima.loc[index_time_min:index_time_max]
    Temperatura_Minima = Temperatura_Minima.loc[index_time_min:index_time_max]
    Flow = Flow.loc[index_time_min:index_time_max]
    
    #Prec_mean = Precipitation.mean(axis=1).values.reshape(-1,1)
    #Prec_var  = Precipitation.var(axis=1).values.reshape(-1,1)
    #Prec_max  = Precipitation.max(axis=1).values.reshape(-1,1)
    #Prec_min  = Precipitation.max(axis=1).values.reshape(-1,1)
    
    #Tmax_mean = Temperatura_Maxima.mean(axis=1).values.reshape(-1,1)
    #Tmax_var  = Temperatura_Maxima.var(axis=1).values.reshape(-1,1)
    #Tmax_max  = Temperatura_Maxima.max(axis=1).values.reshape(-1,1)
    #Tmax_min  = Temperatura_Maxima.max(axis=1).values.reshape(-1,1)
    
    
    #Tmin_mean = Temperatura_Minima.mean(axis=1).values.reshape(-1,1)
    #Tmin_var  = Temperatura_Minima.var(axis=1).values.reshape(-1,1)
    #Tmin_max  = Temperatura_Minima.max(axis=1).values.reshape(-1,1)
    #Tmin_min  = Temperatura_Minima.max(axis=1).values.reshape(-1,1)
    
    
    
    X = np.hstack((Precipitation.values,Temperatura_Maxima.values,Temperatura_Minima.values)).astype(float)
    Y = Flow.values.astype(float)
    
    X = X[~np.isnan(Y).flatten(),:]
    Y = Y[~np.isnan(Y)].reshape(-1,1)
    
    
    return X,Y
    

class REGRESION(object):
    """
    Con esta función se ajusta un modelo de regresión para predecir el caudal.
    
    Datos de Entrada:
    -----------------
    file_Precipitation:        string.  Fichero de precipitación
    file_Temperatura_Maxima:   string.  Fichero de temperatura máxima
    file_Temperatura_Mínima:   string.  Fichero de temperatura mínima
    file_Flow              :   string.  Fichero de aportaciones
    
    name_fit:             string. Nombre con el que se guardan los ajustes.
    path_project:         string. Directorio del proyecto
    n_jobs:               integer. Número de procesadores de computación n_jobs=-1, utilización de la CPU de forma completa
    
    Resultados:
    -----------------
    fit: dictionary. Fichero con los parámetros del ajuste

    """
    def __init__ (self,file_Precipitation, file_Temperatura_Maxima, file_Temperatura_Minima,file_Flow,name_fit,path_project, n_jobs=1):
        
        
        [X,Y] = read_series(file_Precipitation, file_Temperatura_Maxima, file_Temperatura_Minima,file_Flow)
        self.X            = X
        self.Y            = Y
        self.name_fit     = name_fit
        self.path_project = path_project
        self.n_jobs       = n_jobs
        
        
        mask = np.where(~np.isnan(X))
        interp = NearestNDInterpolator(np.transpose(mask), X[mask])
        X = interp(*np.indices(X.shape))
        
        
        self.pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))
        self.pca_pipe.fit(X)
    
        self.Xeof = self.pca_pipe.transform(X)
        
        filename = self.name_fit+'_PCA.sav'
        pickle.dump(self.pca_pipe, open(self.path_project+'/04_REGRESION/'+filename, 'wb'))
    
        
    def fit_regression(self,plot=True):
        """
        Datos de Entrada:
        -----------------
        plot: True o False. Si se quieren plotear los valores del análisis del ajuste.
        
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self.Xeof, self.Y, test_size=0.2,random_state=0)

        
        scoring = {"r2": "r2",  "nse":nse_scorer,"bias":bias_scorer, "explained_variance": "explained_variance","neg_mean_squared_error":"neg_mean_squared_error"}
        #scoring = {"r2": "r2",  "nse":nse_scorer,"bias":bias_scorer}

        self.y_scaler = MinMaxScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled  = self.y_scaler.transform(y_test.reshape(-1, 1))
                
        y_true = y_test  # ✅ valor real, sin escalar
            
        # tuned_parameters = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
        #                     'activation': ['relu','tanh','logistic'],
        #                     'alpha': [0.0001, 0.05],
        #                     'learning_rate': ['constant','adaptive'],
        #                     'solver': ['adam']}

        tuned_parameters = {"hidden_layer_sizes":[
                            (30,), (50,), (80,),(50, 30),
                            # (50, 30), (80, 40), (100, 60),
                            # (60, 40, 20), (100, 50, 25),
                            # (50, 30, 10), (30, 30, 30),
                            # (120, 60, 30), (100, 80, 60, 20),
                            # (150, 100, 50), (200, 100, 50, 25), (256, 128, 64, 32, 16)
                            ],
                            "activation": ["relu"], 
                            "alpha": [1e-2, 1e-1, 1, 10],
                            "solver": ["sgd", "adam"], 
                            'learning_rate': ['constant','adaptive'],
                            "early_stopping": [True],
                            "validation_fraction": [0.2]}

        #self.clf = DaskGridSearchCV(MLPRegressor(max_iter=1000), tuned_parameters, scoring=scoring, refit="r2", n_jobs=self.n_jobs, return_train_score=True) 
        self.clf = GridSearchCV(MLPRegressor(max_iter=3000, random_state=42), tuned_parameters, scoring=scoring, refit="r2", n_jobs=self.n_jobs, return_train_score=True)    
        with joblib.parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            self.clf.fit(X_train, y_train.ravel())


        print("Best parameters set found on development set:")
        print()
        print(self.clf.best_params_)

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        y_pred = self.clf.predict(X_test)
        #y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = y_test


        filename = self.name_fit+'_ANN.sav'
        pickle.dump(self.clf, open(self.path_project+'/04_REGRESION/'+filename, 'wb'))
        pickle.dump(self.y_scaler, open(self.path_project+'/04_REGRESION/'+self.name_fit+'_yScaler.sav', 'wb'))
        np.savetxt(self.path_project+'/04_REGRESION/'+'y_test_'+self.name_fit+'.csv', y_test, delimiter=',')
        np.savetxt(self.path_project+'/04_REGRESION/'+'x_test_'+self.name_fit+'.csv', X_test, delimiter=',')

        y_pred = self.clf.predict(X_test)
        #y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = y_test.ravel()
        
        y_pred[y_pred<0]=0
        nse  = he.evaluator(he.nse, y_pred, y_true)[0]
        bias = he.evaluator(he.pbias, y_pred, y_true)[0]
        r2   = rsquared(y_pred, y_true)


        fit = np.polyfit(np.ravel(y_true), np.ravel(y_pred), 1)
        fit_fn = np.poly1d(fit)

        fig, ax = plt.subplots(figsize=(8, 8))


        ax.plot(y_true, y_pred, '.r')
        ax.plot(y_true, fit_fn(y_true), '-b', linewidth=0.8)
        ax.set_ylabel('Simulado', fontsize=20)
        ax.set_xlabel('Observado', fontsize=20)

        m1=max(y_true)
        m2=max(y_pred)
        mT=max(m1, m2)

        ax.axis('equal')
        ax.set_xlim(0, mT)
        ax.set_ylim(0, mT)
        ax.text(mT/1.3, mT/3.5, 'Bias = '"{0:.2f}".format(np.round(bias,2)), ha="center", va="center",
                          fontdict=dict(fontsize=23, fontweight='bold', color="black"), 
                          bbox=dict(facecolor="white", alpha=0.85))
        ax.text(mT/1.3, mT/3.5-mT*0.1, 'R\u00b2 = '"{0:.2f}".format(np.round(r2,2)), ha="center", va="center",
                          fontdict=dict(fontsize=23, fontweight='bold', color="black"), 
                          bbox=dict(facecolor="white", alpha=0.85))
        ax.text(mT/1.3, mT/3.5-mT*0.2, 'NSE = '"{0:.2f}".format(np.round(nse,2)), ha="center", va="center",
                          fontdict=dict(fontsize=23, fontweight='bold', color="black"), 
                          bbox=dict(facecolor="white", alpha=0.85))
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
        ax.tick_params(labelsize=20)
            
        if plot==False:
            plt.close(fig)
           
        fig.savefig(self.path_project+'/07_INFORME/Figuras/'+'Validacion_Modelo.png',bbox_inches='tight')
            
        return self.clf
    
    
    def predict(self):
        Sim_scaled = self.clf.predict(self.Xeof)
        Sim = self.y_scaler.inverse_transform(Sim_scaled.reshape(-1, 1)).ravel()
        return Sim
    
    
class SIM_REGRESION(object):   
    def __init__ (self,file_model_Reg,file_model_PCA,file_yScaler, prec, tmax, tmin):
        
        self.file_model_Reg = file_model_Reg
        self.file_model_PCA = file_model_PCA
        self.file_yScaler   = file_yScaler
        self.prec           = prec
        self.tmax           = tmax
        self.tmin           = tmin
          
    
    def simulation(self):
        
        Model_Reg = pickle.load(open(self.file_model_Reg, 'rb'))
        Model_PCA = pickle.load(open(self.file_model_PCA, 'rb'))
        y_scaler  = pickle.load(open(self.file_yScaler, 'rb'))

        Prec_  = self.prec.dropna()
        Tmax_  = self.tmax.dropna()
        Tmin_  = self.tmin.dropna()
        concat_ = pd.concat((Prec_,Tmax_,Tmin_),axis=1)
        concat_ = concat_.dropna()
        n_p = len(Prec_.columns)
        
        self.Aportaciones = pd.DataFrame(index=concat_.index ,columns = ['Hm3/mes'])
        

        X = np.hstack((concat_.iloc[:,0:n_p].values,concat_.iloc[:,n_p:n_p*2].values,concat_.iloc[:,n_p*2:n_p*3].values)).astype(float)
        X_eof = Model_PCA.transform(X)
        Sim = Model_Reg.predict(X_eof).reshape(-1, 1)
        #Sim = y_scaler.inverse_transform(Sim_scaled)
        self.Aportaciones.loc[concat_.index, :] = Sim
        self.Aportaciones[self.Aportaciones<0] = 0

       
    def validation_sim(self,aporta_real_time_serie,path_output):

        flow_real = aporta_real_time_serie.dropna()
        flow_sim  = self.Aportaciones.dropna()
        concat_ = pd.concat((flow_real,flow_sim),axis=1)
        concat_ = concat_.dropna()

        plot_val_serie(concat_.iloc[:,1], concat_.iloc[:,0],path_output)
        plot_curve_CC(concat_.iloc[:,1],concat_.iloc[:,0],path_output)


    def save_series(self,name_file,path_output):
        self.Aportaciones.to_csv(path_output+name_file+'.csv')
        
        
        
    
                  
