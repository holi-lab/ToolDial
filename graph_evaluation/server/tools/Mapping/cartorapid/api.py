import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vector_tile(y: str, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns Vector tiles as PBF. 
		Vector tilles are tiles that can be used with mapping libraries such as Maplibre."
    
    """
    url = f"https://cartorapid.p.rapidapi.com/data/france-vector/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartorapid.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_raster_tile(y: str, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns Raster tiles as PNG. 
		Raster tilles are static image tiles that can be used with mapping libraries such as Leaflet."
    
    """
    url = f"https://cartorapid.p.rapidapi.com/styles/basic/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cartorapid.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

