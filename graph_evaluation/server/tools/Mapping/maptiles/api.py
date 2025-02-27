import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmaptilewithspanishlabels(z: int, x: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raster Map Tiles with Spanish Labels. Please see [tutorial](https://rapidapi.com/MapTilesApi/api/maptiles/tutorials/openstreetmap-in-spanish---add-a-spanish-map-to-your-website-with-leaflet-js-and-maptiles-api) on how to use the Spanish world map."
    z: zoom (from 0 up to zoom 19)
        x: X-number of tile (see documentation)
        y: Y-number of tile (see documentation)
        
    """
    url = f"https://maptiles.p.rapidapi.com/es/map/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmaptilewithenglishlabels(x: int, z: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raster Map Tiles with English Labels. Please see our [tutorials page](https://rapidapi.com/MapTilesApi/api/maptiles/tutorials) on how to use this."
    x: X-number of tile (see documentation)
        z: zoom (from 0 up to zoom 19)
        y: Y-number of tile (see documentation)
        
    """
    url = f"https://maptiles.p.rapidapi.com/en/map/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmaptilewithfrenchlabels(x: int, y: int, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raster Map Tiles with French Labels"
    x: X-number of tile (see documentation)
        y: Y-number of tile (see documentation)
        z: zoom (from 0 up to zoom 19)
        
    """
    url = f"https://maptiles.p.rapidapi.com/fr/map/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandardmaptile(y: int, x: int, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standard tiles with labels in local language for a place (untranslated and untransliterated) as known from OpenStreetMap"
    y: Y-number of tile (see documentation)
        x: X-number of tile (see documentation)
        z: zoom (from 0 up to zoom 19)
        
    """
    url = f"https://maptiles.p.rapidapi.com/local/osm/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

