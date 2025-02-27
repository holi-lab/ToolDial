import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def datos_hist_ritcos_nacionales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos, desde el 7 de marzo hasta el día de hoy, de contagiados confirmados, muertes y "recuperados"  Estos datos corresponden al total a nivel país."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/historical/nation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_hist_ricos_regionales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos históricos de contagiados y muertes por región. En el caso de los contagiados, los datos están disponibles desde el 7 de marzo hasta el día de hoy. Para el caso de las muertes, los datos están sólo disponibles desde el 1 de abril, que es desde en donde el Gobierno tiene publicadas las cifras oficiales."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/historical/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_hist_ricos_regionales_para_una_regi_n_en_espec_fico(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos de contagiados y muertes por región de la misma forma que en el endpoint anterior, sin embargo, si se entrega el parámetro de region-id (sin los corchetes) se reciben los datos para la región indicada. Los códigos de las regiones corresponden al antiguo sistema de regiones numeradas (e.g, Metropolitana es 13.) Se pueden consultar todos los códigos de regiones disponibles utilizando el Endpoint de models que se detalla más abajo."
    id: ID de la región a consultar. 13 es RM.
        
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/historical/regions"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_hist_ricos_comunales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos históricos que el Gobierno ha publicado para todas las comunas. Estos datos son entregados por parte del ministerio con poca frecuencia, usualmente cada un par de días o un poco más. Para este caso, sólo están disponibles los contagiados confirmados, que es lo único que el gobierno ha entregado de forma oficial. Usualmente un 0 representa que no hay datos significativos para esta comuna."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/historical/communes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_hist_ricos_comunales_para_una_comuna_en_espec_fico(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos de la misma forma que en el endpoint anterior, sin embargo, al igual que con las regiones, se puede obtener para una comuna en específico, entregando el commune-id, sin los corchetes. La lista de los IDs de comuna está disponible en el endpoint de models de más abajo."
    id: ID de la comuna a consultar. 5101 es Valparaíso
        
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/historical/communes?id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_del_ltimo_d_a_para_el_pa_s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtienen los datos de contagiados, muertes y recuperados para el último día en que el gobierno ha realizado un reporte. El gobierno actualmente realiza un reporte todas las mañanas, con los datos cortados hasta el día anterior a las 21:00 hrs."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/latest/nation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_del_ltimo_d_a_para_todas_las_regiones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datos de regiones para el último día reportado por el ministerio. Con este endpoint se obtiene el último reporte para todas las regiones."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/latest/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_del_ltimo_d_a_para_la_comuna_seleccionada(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datos de comunas para el último día reportado por el ministerio. Con este endpoint se obtiene el último reporte para la comuna señalada en el commune-id."
    id: ID de la comuna a consultar. 13 es RM.
        
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/latest/communes"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_del_ltimo_d_a_para_todas_las_comunas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datos de comunas para el último día reportado por el ministerio. Con este endpoint se obtiene el último reporte para todas las comunas."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/latest/communes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datos_del_ltimo_d_a_para_la_regi_n_seleccionada(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Datos de regiones para el último día reportado por el ministerio. Con este endpoint se obtiene el último reporte para la región señalada en el region-id."
    id: ID de la región a consultar. 13 es RM.
        
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/latest/regions"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regiones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtiene una lista de todas las regiones junto a su ID respectivo."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/models/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comunas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Se obtiene una lista con todas las comunas del país, junto a su ID y a qué región del país pertenecen."
    
    """
    url = f"https://chile-coronapi1.p.rapidapi.com/v3/models/communes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chile-coronapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

