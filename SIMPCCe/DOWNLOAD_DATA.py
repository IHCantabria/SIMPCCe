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
import zipfile


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
    print('--- Descargando datos de cambio climático AEMET ---')
    var = ['PRCPTOT', 'TXMM', 'TNMM']
    escenarios = ['HIST', 'RCP45', 'RCP85']

    # Crear estructura de carpetas si no existen
    base_path = os.path.join(path_output, 'AEMET/CAMBIO_CLIMATICO/')
    os.makedirs(base_path, exist_ok=True)

    for v in var:
        var_path = os.path.join(base_path, v)
        os.makedirs(var_path, exist_ok=True)
        for sc in escenarios:
            sc_path = os.path.join(var_path, sc)
            os.makedirs(sc_path, exist_ok=True)

            # Definir URL y nombre del archivo ZIP
            url_CC_AEMET = 'http://www.aemet.es/documentos_d/serviciosclimaticos/cambio_climat/datos_mensuales/'
            zip_file = f'R_{v}_{sc}_COR_SIG.zip'
            url = url_CC_AEMET + zip_file
            local_zip_path = os.path.join(sc_path, zip_file)

            # Verificar si ya existen archivos descomprimidos en la carpeta destino
            if any(file.endswith('.txt') for file in os.listdir(sc_path)):
                print(f'Archivos descomprimidos ya existen en: {sc_path}. Se omite la descarga.')
                continue

            print(f'Descargando variable: {v}, Escenario: {sc}')
            not_downloaded = True
            nnn = 0

            while not_downloaded:
                try:
                    print(f'Descargando desde: {url}')
                    # Descargar archivo
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_zip_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)

                    # Descomprimir el archivo y eliminar el ZIP
                    shutil.unpack_archive(local_zip_path, sc_path)
                    os.remove(local_zip_path)  # Eliminar archivo ZIP descargado
                    print(f'Archivo {zip_file} descargado y descomprimido correctamente.')
                    not_downloaded = False

                except Exception as e:
                    print(f"Fallo de conexión: {e}. Reintentando...")
                    nnn += 1
                    if nnn > 3:
                        print(f'Error: No se pudo descargar el archivo {zip_file}. Verifique su conexión.')
                        break

    print('--- Descarga y extracción completadas ---')

                            
                            
import os
import requests
import patoolib
import platform
import time

def _download_SIMPA(path_output):
    sistema_operativo = platform.system()
    print('--- Descargando datos de Aportaciones SIMPA ---')

    # Crear directorios necesarios
    aportaciones_path = os.path.join(path_output, 'SIMPA', 'Aportaciones')
    os.makedirs(aportaciones_path, exist_ok=True)

    # Rango de años para los archivos
    for i in range(1940, 2016, 10):
        # Construir URL según el año
        if i == 2010:
            url = f'http://ceh-flumen64.cedex.es/descargas/ERH_Entregamayo2019B/Aportacion_acumulada_{str(i)[-2:]}_{str(i+6)[-2:]}.rar'
        else:
            url = f'http://ceh-flumen64.cedex.es/descargas/ERH_Entregamayo2019B/Aportacion_acumulada_{str(i)[-2:]}_{str(i+10)[-2:]}.rar'

        # Verificar si el archivo ya ha sido procesado
        asc_file = os.path.join(aportaciones_path, f'acaesh{i}_10.asc')
        if os.path.exists(asc_file):
            print(f'Archivo {asc_file} ya existe. Se omite la descarga.')
            continue

        # Nombre del archivo descargado
        file_name = os.path.join(aportaciones_path, os.path.basename(url))
        download_successful = False

        # Descargar el archivo con múltiples intentos
        for attempt in range(1, 6):
            try:
                print(f'Descargando {url} (Intento {attempt})...')
                with requests.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    with open(file_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f'Archivo {file_name} descargado correctamente.')
                download_successful = True
                break
            except requests.RequestException as e:
                print(f'Error en la descarga: {e}')
                time.sleep(5)

        if not download_successful:
            print(f'No se pudo descargar el archivo {url}. Verifique su conexión a internet.')
            continue

        # Extraer el archivo descargado usando patoolib
        try:
            print(f'Extrayendo {file_name} con patoolib...')
            patoolib.extract_archive(file_name, outdir=aportaciones_path)
            print(f'Archivo {file_name} extraído correctamente.')
        except Exception as e:
            print(f'Error al extraer {file_name}: {e}')
            continue

        # Eliminar el archivo descargado tras la extracción
        try:
            os.remove(file_name)
            print(f'Archivo {file_name} eliminado tras la extracción.')
        except OSError as e:
            print(f'Error al eliminar el archivo {file_name}: {e}')

    print('*********** Descarga y extracción finalizadas **************')


def _descarga_datos_SPAIN02(path_output):
    os.chdir(path_output)
    # Comprobar si ya existen archivos descargados
    if os.path.exists('./AEMET/SPAIN02/Spain02_v5.0_MM_010reg_aa3d_pr.nc'):
        print('#### Ficheros Existentes ####')
    else:
        # Nueva URL base para la descarga de datos mensuales desde AEMET
        url_base = 'https://www.aemet.es/documentos/es/serviciosclimaticos/cambio_climat/datos_diarios/dato_observacional/rejilla_20km/v5/'
        archivos = [
            'Spain02_v5.0_DD_010reg_aa3d_tasmin.nc.zip',
            'Spain02_v5.0_DD_010reg_aa3d_tasmax.nc.zip',
            'Spain02_v5.0_DD_010reg_aa3d_pr.nc.zip',
        ]

        os.makedirs('./AEMET/', exist_ok=True)
        os.makedirs('./AEMET/SPAIN02/', exist_ok=True)

        for archivo in archivos:
            nombre_archivo = os.path.basename(archivo)
            local_file = os.path.join('./AEMET/SPAIN02/', nombre_archivo)

            if os.path.exists(local_file.replace('.zip', '')):
                print(f'Archivo {nombre_archivo.replace(".zip", "")} ya existe. Se omite la descarga.')
                continue

            url = url_base + archivo
            print(f'Descargando {nombre_archivo} desde {url}...')

            # Realiza la solicitud HTTP para descargar el archivo
            with requests.get(url, stream=True, verify=False) as r:
                if r.status_code != 200:
                    print(f"Error al descargar {nombre_archivo}: Estado {r.status_code}")
                    continue

                with open(local_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4096 * 64):
                        f.write(chunk)
            print(f'Archivo {nombre_archivo} descargado correctamente.')

            # Extraer el archivo ZIP directamente a la carpeta de destino
            print(f'Extrayendo {nombre_archivo}...')
            with zipfile.ZipFile(local_file, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, './AEMET/SPAIN02/')
                    print(f'Extraído: {file}')

            # Eliminar el archivo ZIP
            os.remove(local_file)
            print(f'Archivo {nombre_archivo} eliminado correctamente.')

        print('*********** Descarga y extracción finalizadas **************')

            
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
            
    