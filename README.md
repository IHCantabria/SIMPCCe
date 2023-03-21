[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7009927.svg)](https://doi.org/10.5281/zenodo.7009927)

# SIMPCCe: Simulador de Pronósticos de Cambio Climático en Embalses

<p float="left">
<img src="https://ihcantabria.com/wp-content/uploads/2020/07/Logo-IHCantabria-Universidad-Cantabria-cmyk.png" alt="drawing" width="240"/>
</p>

La librería SIMPCCe se desarrolla como complemento a la **GUÍA METODOLÓGICA PARA LA ESTIMACIÓN DE APORTACIONES MÍNIMAS A EMBALSES EN EL CONTEXTO DE CAMBIO CLIMÁTICO**

## Aplicación SIMPCCe
Además de la creación de la librería SIMPCCe se ha desarrollado una aplicación con el mismo nombre que permite aplicar la metodología descrita en la guía sin necesidad de la ejecución del código ni instalación de los diferentes requerimientos. Esta aplicación es un ejecutable en la que el usuario interactua con ella para configurar el proyecto en el punto deseado. Para obtener la aplicación vaya a la sección de **Releases** de esta página web (en la parte derecha de la página) y descargue el archivo ejecutable o pulse en el siguiente enlace [**SIMPCCe.exe**](https://github.com/IHCantabria/SIMPCCe/releases/download/v1.0.0/SIMPCCe-Windows.exe)

<p float="left">
<img src="https://github.com/IHCantabria/SIMPCCe/blob/main/SIMPCCe.png" alt="drawing" width="400"/>
</p>

**La idea fundamental de la librería y la aplicación es el uso complementario a la guía; por tanto, para su utilización es necesario que el usuario conozca la metodología descrita en la guía para evitar un uso incorrecto.**

## Contenido
| Directorio | Contenido |
| :-------- | :------- |
|  [SIMPCCe](https://github.com/IHCantabria/SIMPCCe/tree/main/SIMPCCe) | Código de Python donde se han implementado la librería que permite realizar todo el análisis definido en la metodología.
| [doc](https://github.com/IHCantabria/SIMPCCe/tree/main/doc) | Directorio donde se localiza la guía metodológica y el manual de la aplicación SIMPCCe.
| [notebooks](https://github.com/IHCantabria/SIMPCCe/tree/main/notebooks) |  Jupyter notebooks donde se realiza un ejemplo de aplicación y que puede ser utilizado para realizar cualquier estudio.
| [app](https://github.com/IHCantabria/SIMPCCe/releases/download/v1.0.0/SIMPCCe-Windows.exe) |  Enlace de descarga de la aplicación

## Requerimientos

Los scripts y cuadernos (jupyter) se proporcionan en [Python](https://www.python.org/) tara asegurar la reproducibilidad y reutilización de los resultados. La forma más sencilla de cumplir con todos estos requisitos es utilizando un entorno dedicado de [conda](https://docs.conda.io) , que se puede instalar fácilmente mediante la ejecución de la siguientes líneas de comando y la descarga o el clonado de la librería para acceder al [fichero  yml](https://github.com/IHCantabria/SIMPCCe/blob/main/environment.yml) que permitirá instalar todas las librerías necesaria para la utilización de la librería SIMPCCe

### ¿Cómo puedo instalar python en mi equipo?

La instalación de Python, el Notebook y todos los paquetes que utilizaremos, por separado puede ser una tarea ardua y agotadora, pero no se preocupe: ¡alguien ha hecho ya el trabajo duro!

__[Anaconda](https://continuum.io/anaconda/) es una distribución de Python que recopila muchas de las bibliotecas necesarias en el ámbito de la computación científica__ y desde luego, todas las que necesitaremos en este curso. Además __incluye herramientas para programar en Python, como [Jupyter Notebook](http://jupyter.org/) o [Spyder](https://github.com/spyder-ide/spyder#spyder---the-scientific-python-development-environment)__ (un IDE al estilo de MATLAB).

Lo único que necesita hacer es:

* Ir a la [página de descargas de Anaconda](http://continuum.io/downloads).
* Seleccionar tu sistema operativo (Windows, OSX, Linux).
* Descargar Anaconda (utilizaremos Python 3.X).

 Puede seguir los pasos a través de este [video](https://youtu.be/x4xegDME5C0?list=PLGBbVX_WvN7as_DnOGcpkSsUyXB1G_wqb).

Una vez instalado [Anaconda](https://continuum.io/anaconda/) y python, a través de [Anaconda Promnt](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) instalado en el equipo se escriben las siguiente líneas: 

```sh
conda env create -f environment.yml
conda activate SIMPCCe_env
```
Tras crear el entorno de Conda específico para esta librería se procede a la instalación.
Si se ha clonado o descargado la librería en el equipo desde Github, desde la carpeta donde se se encuentra el fichero [setup.py](https://github.com/IHCantabria/SIMPCCe/blob/main/setup.py) se ejecuta la siguiente línea
```sh
pip install -e. SIMPCCe
```
También puede ser instalado directamente desde Github:
```sh
pip install git+https://github.com/IHCantabria/SIMPCCe
```
## Ejemplo de uso

Los ejemplos de uso de la librería `SIMPCCe` están disponibles en forma de [cuadernos jupyter](https://github.com/IHCantabria/SIMPCCe/tree/main/notebooks). Para ejecutar los ejemplos siga los siguientes pasos:

1. Descargue la carpeta [notebooks](https://github.com/IHCantabria/SIMPCCe/tree/main/notebooks) desde el repositorio de github, o navegue hasta la carpeta si ha clonado el repo.
2. Abre el cuaderno Jupyter de Jupyter Lab (escribe `jupyter notebook` o `jupyter lab` en el terminal)
3. Abra la prueba disponible en la carpeta  [Aplicación_SIMPCCe](https://github.com/IHCantabria/SIMPCCe/blob/main/notebooks/Aplicaci%C3%B3n_SIMPCCe.ipynb)

<font color='red'><font color='red'>**Es importante que donde se ejecute el [notebook](https://github.com/IHCantabria/SIMPCCe/blob/main/notebooks/Aplicaci%C3%B3n_SIMPCCe.ipynb) se localice la carpeta [data](https://github.com/IHCantabria/SIMPCCe/tree/main/notebooks/data) e [images](https://github.com/IHCantabria/SIMPCCe/tree/main/notebooks/images) para visualizarlo y ejecutarlo correctamente**

## Colaboradores
La versión original de la librería y aplicación ha sido desarrollada por:

+ Salvador Navas
+ Manuel del Jesus

## Licencia

Este programa es software libre: puede redistribuirlo y/o modificarlo bajo los términos de la Licencia Pública General GNU publicada por la Fundación para el Software Libre, ya sea la versión 3 de la Licencia, o (a su elección) cualquier versión posterior.

Este programa se distribuye con la esperanza de que sea útil pero SIN NINGUNA GARANTÍA; ni siquiera la garantía implícita de COMERCIALIZACIÓN o ADECUACIÓN A UN PROPÓSITO PARTICULAR.  Consulte la Licencia Pública General de GNU para más detalles.

Debería haber recibido una copia de la Licencia Pública General GNU junto con este programa.  Si no es así, consulte <https://www.gnu.org/licenses/>.
