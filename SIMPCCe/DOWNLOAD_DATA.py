'''
La librería contiene las funciones que permiten descargar los datos climáticos históricos y 
futuros de AEMET además de las aportaciones de SIMPA
	Autores: 
	    + Salvador Navas Fernández
        + Manuel del Jesus
'''


import os
import shutil
import urllib.request
import sys
import requests
import time
import patoolib
import yaml


def download_data(path_output,url):
    headers = requests.head(url, headers={'accept-encoding': ''}).headers
    r = requests.get(url, allow_redirects=True, stream=True,  verify=False,timeout=30)
    file_size = int(headers['Content-Length'])
    
    if file_size<=216:
        file_size = 500000000
    downloaded = 0
    start = last_print = time.time()
    #name = path_output+v+'/'+sc+'/'+'R_'+v+'_'+sc+'_COR_SIG.zip'
    with open(path_output, 'wb') as fp:
        for chunk in r.iter_content(chunk_size=4096 * 64):
            downloaded += fp.write(chunk)
            now = time.time()
            if now - last_print >= 1:
                pct_done = round(downloaded / file_size * 100)
                speed = round(downloaded / (now - start) / 1024)
                print(f"Download {pct_done} % done, avg speed {speed} kbps")
                last_print = time.time()
                
def _download_AEMET_CC(path_output):
        print('---Descargando datos de cambio climático AEMET---')
        import requests
        var = ['PRCPTOT','TXMM','TNMM']
        escenarios = ['HIST','RCP45','RCP85']
        if os.path.exists(path_output+'/AEMET/CAMBIO_CLIMATICO/')==False:
            os.makedirs(path_output+'/AEMET/CAMBIO_CLIMATICO/')
        for v in var:
            if os.path.exists(path_output+'/AEMET/CAMBIO_CLIMATICO/'+v+'/')==False:
                os.makedirs(path_output+'/AEMET/CAMBIO_CLIMATICO/'+v+'/',exist_ok=True)
            for sc in escenarios:
                if os.path.exists(path_output+'/AEMET/CAMBIO_CLIMATICO/'+v+'/'+sc+'/')==False:
                    os.makedirs(path_output+'/AEMET/CAMBIO_CLIMATICO/'+v+'/'+sc+'/',exist_ok=True)

                url_CC_AEMET = 'http://www.aemet.es/documentos_d/serviciosclimaticos/cambio_climat/datos_mensuales/'
                url = url_CC_AEMET+'R_'+v+'_'+sc+'_COR_SIG.zip'
                name = path_output+'/AEMET/CAMBIO_CLIMATICO/'+v+'/'+sc+'/'+'R_'+v+'_'+sc+'_COR_SIG.zip'
                print('Descargando variable: '+v +' Ecenario: '+sc)
                not_downloaded = True
                nnn=0
                while not_downloaded:
                    try:
                        print(url)
                        #urllib.request.urlretrieve(url0,path_output+'/'+var+'/'+var+'_day_BCSD_'+rcp+'_r1i1p1_'+mod+'_'+str(t)+'.nc')
                        download_data(name,url)
                        shutil.unpack_archive(name, path_output+'/AEMET/CAMBIO_CLIMATICO/'+'/'+v+'/'+sc+'/')
                        os.remove(name)
                        not_downloaded = False
                    except:
                        print("Fallo de conexión...Reestableciendo")
                        nnn=nnn+1
                        if nnn>3:
                            print('Compruebe su conexión a internet, no es posible realizar la descarga')
                            break
                            
                            
def _download_SIMPA(path_output):
    import requests
    import patoolib
    import platform
    sistema_operativo = platform.system()
    print('---Descargando datos de Aportaciones SIMPA---')

    if os.path.exists(path_output+'/SIMPA/Aportaciones/')==False:
        os.makedirs(path_output+'/SIMPA/',exist_ok=True)
        os.makedirs(path_output+'/SIMPA/Aportaciones/',exist_ok=True)

    for i in range(1940,2016,10):
        if i==2010:
            url = 'http://ceh-flumen64.cedex.es/descargas/ERH_Entregamayo2019B/Aportacion_acumulada_'+str(i)[-2:]+'_'+str(i+6)[-2:]+'.rar'
        else:
            url = 'http://ceh-flumen64.cedex.es/descargas/ERH_Entregamayo2019B/Aportacion_acumulada_'+str(i)[-2:]+'_'+str(i+10)[-2:]+'.rar'



        if os.path.exists(path_output+'/SIMPA/Aportaciones/'+'acaesh'+str(i)+'_10.asc'):
                continue
        else:
            name = path_output+'/SIMPA/Aportaciones/'+url.split('/')[-1]

            downloaded = 0
            start = last_print = time.time()
            not_downloaded = True
            nnn=0
            while not_downloaded:
                try:
                    print(url)
                    #urllib.request.urlretrieve(url0,path_output+'/'+var+'/'+var+'_day_BCSD_'+rcp+'_r1i1p1_'+mod+'_'+str(t)+'.nc')
                    download_data(name,url)
                    not_downloaded = False
                except:
                    print("Fallo de conexión...Reestableciendo")
                    nnn=nnn+1
                    if nnn>10:
                        print('Compruebe su conexión a internet, no es posible realizar la descarga')
                        break
            #shutil.unpack_archive(name, path_output+'/SIMPA/Aportaciones/')
            #os.remove(name)
            try:
                if sistema_operativo=='Windows':
                    if os.system(sys._MEIPASS+'/7z/7z.exe e -y '+name+' -o'+path_output+'/SIMPA/Aportaciones/')!=0:
                        raise ValueError('Verifique que tiene todos los permisos en el equipo')   
                else:
                    patoolib.extract_archive(name, outdir=path_output+'/SIMPA/Aportaciones/')
            except:
                raise ValueError('Verifique si en su equipo está instalado algún programa para descomprimir ficheros RAR')
                raise ValueError('Verifique si el equipo tiene suficiente espacio')
        os.remove(name)
    print('***********Descarga finalizada**************')



def _descarga_datos_SPAIN02(path_output):
    os.chdir(path_output)
    import patoolib
    if os.path.exists('./AEMET/SPAIN02/Spain02_v5.0_MM_010reg_aa3d_pr.nc'):
        print('#### Ficheros Existentes ####')
    else:
        url = 'http://meteo.unican.es/work/datasets/Spain02_v5.0_010reg_aa3d.tar.gz'
        os.makedirs('./AEMET/',exist_ok=True)
        os.makedirs('./AEMET/SPAIN02/',exist_ok=True)
        headers = requests.head(url, headers={'accept-encoding': ''}).headers
        r = requests.get(url, allow_redirects=True, stream=True,  verify=False)
        file_size = int(headers['Content-Length'])
        downloaded = 0
        start = last_print = time.time()
        name  = './AEMET/SPAIN02/'+url.split('/')[-1]
        with open(name, 'wb') as fp:
            for chunk in r.iter_content(chunk_size=4096 * 64):
                downloaded += fp.write(chunk)
                now = time.time()
                if now - last_print >= 1:
                    pct_done = round(downloaded / file_size * 100)
                    speed = round(downloaded / (now - start) / 1024)
                    print(f"Download {pct_done} % done, avg speed {speed} kbps")
                    last_print = time.time()
        patoolib.extract_archive(name, outdir='./AEMET/SPAIN02/')
        allfiles = os.listdir('./AEMET/SPAIN02/Spain02_v5.0_010reg_aa3d/')

        for f in allfiles:
            shutil.move('./AEMET/SPAIN02/Spain02_v5.0_010reg_aa3d/' + f, './AEMET/SPAIN02/' + f)
        os.remove('./AEMET/SPAIN02/Spain02_v5.0_010reg_aa3d.tar.gz')
        os.rmdir('./AEMET/SPAIN02/Spain02_v5.0_010reg_aa3d/')
        print('***********Descarga finalizada**************')
            
            
            
def descarga_AEMET_SIMPA(path_output,path_project,name_project):
    """
    Con esta función se procede a descargar toda la información climática y de aportaciones necesaria.
    Datos de Entrada:
    -----------------
    path_output:  string. Directorio donde guardar la información descargada
    path_project: string. Directorio del proyecto
    name_project: string. Nombre del proyecto

    Resultados:
    -----------------
    Conjunto de carpetas y ficheros en la carpeta definida como output.

    """
    print('##### Descargando datos climáticos AEMET - SPAIN02')
    _descarga_datos_SPAIN02(path_output)
    print('##### Descargando aportaciones SIMPA')
    _download_SIMPA(path_output)
    print('##### Descargando datos cambio climático AEMET')
    _download_AEMET_CC(path_output)
    
    file_yml = path_project+'/'+name_project+'.yml'

    with open(file_yml) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['path_datos'] = path_output

    with open(file_yml, 'w') as file:
        documents = yaml.dump(params, file)
            
    