import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retinatile(y: int, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a 512px times 512px retina tile. Please see the [about page](https://rapidapi.com/MapTilesApi/api/retina-tiles/details) or  the [tutorials](https://rapidapi.com/MapTilesApi/api/retina-tiles/tutorials) on how to use this Endpoint (yep, the generic RapidAPI code examples here are not helpful for maps)."
    y: y tile number
        z: Zoom factor between 0 and 19
        x: x tile number
        
    """
    url = f"https://retina-tiles.p.rapidapi.com/local/osm@2x/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "retina-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standardtile(y: int, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a standard 256px times 256px raster tile. Please see the [about page](https://rapidapi.com/MapTilesApi/api/retina-tiles/details) or the [tutorials](https://rapidapi.com/MapTilesApi/api/retina-tiles/tutorials) on how to use this Endpoint (yep, the generic RapidAPI code examples here are not helpful for maps)."
    y: y tile number
        z: Zoom factor between 0 and 19
        x: x tile number
        
    """
    url = f"https://retina-tiles.p.rapidapi.com/local/osm/v1/{z}/{x}/{y}.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "retina-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

