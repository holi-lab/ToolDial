import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def puntos_de_venta(layer: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    layer: Recolecta todos los registros de Puntos de Venta georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mercados(layer: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    layer: Recolecta todos los registros de Mercados georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productores(layer: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    layer: Recolecta todos los registros de productores georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def todo(layer: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    layer: Recolecta todos los registros georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bares(layer: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    layer: Recolecta todos los registros de bares georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ferias(layer: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recolecta"
    layer: Recolecta todos los registros de ferias georeferenciados
        
    """
    url = f"https://ecoalimentate-v1.p.rapidapi.com/wp-content/plugins/leaflet-maps-marker/leaflet-geojson.php"
    querystring = {'layer': layer, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoalimentate-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

