'''
La librería contiene las funciones que permiten configuran la distribución de carpetas necesarias 
para llevar a cabo un proyecto
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''

import os
import shutil
import yaml
import glob
def generate_project(path_project,name_project):
        """
        Con esta función se configuran los directorios necesarios para la ejecución de un nuevo proyecto.
        Datos de Entrada:
        -----------------
        path_project: string. Directorio del proyecto
        name_project: string. Nombre del proyecto

        Resultados:
        -----------------
        Conjunto de carpetas donde se irán guardando los ficheros resultantes de los procesos metodológicos.

        """
    
        shutil.copyfile('./data/File_Project.yml',path_project+'/'+name_project+'.yml') #sys._MEIPASS
        os.makedirs(path_project+'/01_CLIMA/',exist_ok=True)
        #os.makedirs(path_project+'/01_CLIMA/Precipitacion/',exist_ok=True)
        #os.makedirs(path_project+'/01_CLIMA/Temperatura/',exist_ok=True)
        os.makedirs(path_project+'/02_GIS/',exist_ok=True)
        os.makedirs(path_project+'/03_APORTACIONES/',exist_ok=True)
        os.makedirs(path_project+'/04_REGRESION/',exist_ok=True)
        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/',exist_ok=True)
        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/',exist_ok=True)
        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX/',exist_ok=True)
        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/01_CLIMA/CORDEX_BIAS_CORRECTED/',exist_ok=True)
        os.makedirs(path_project+'/05_CAMBIO_CLIMATICO/02_APORTACIONES/',exist_ok=True)
        os.makedirs(path_project+'/06_ANALISIS_RESULTADOS/',exist_ok=True)
        os.makedirs(path_project+'/07_INFORME/',exist_ok=True)
        os.makedirs(path_project+'/07_INFORME/Figuras/',exist_ok=True)
        
        for filename in glob.glob(os.path.join('./data/Capas_Hidro/', '*.*')):
            shutil.copy(filename, path_project+'02_GIS/')
            
        for filename in glob.glob(os.path.join('./data/Plantillas/', '*.*')):
            shutil.copy(filename, path_project+'07_INFORME/Figuras/')
        

        file_yml = path_project+'/'+name_project+'.yml'

        with open(file_yml) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        params['path_project'] = path_project
        params['name_project'] = name_project

        with open(path_project+'/'+name_project+'.yml', 'w') as file:
            documents = yaml.dump(params, file)
    
    
        