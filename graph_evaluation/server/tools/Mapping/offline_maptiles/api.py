import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def local_osm_v1_z_x_y_png(y: int, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download OpenStreetMap standard tile"
    y: y tile number
        z: Zoom factor between 0 and 19
        x: x tile number
        
    """
    url = f"https://offline-maptiles.p.rapidapi.com/local/osm/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "offline-maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def en_map_v1_z_x_y_png(z: int, y: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download English tile"
    z: Zoom factor between 0 and 19
        y: y tile number
        x: x tile number
        
    """
    url = f"https://offline-maptiles.p.rapidapi.com/en/map/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "offline-maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

