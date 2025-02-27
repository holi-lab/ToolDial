import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def map_script(host: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a JavaScript file to control the tile map that runs on the browser."
    host: A domain name of your website.
        
    """
    url = f"https://navitime-maps.p.rapidapi.com/map_script"
    querystring = {'host': host, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-maps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def map_image_get(width: int, center: str, zoom: int, height: int, datum: str='wgs84', coord_unit: str='degree', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a static map image of the location, range and zoom value specified in the parameter."
    width: Width of the map.
(Minimum value: 1, Maximum value: 999)
        center: Center of the map.
        zoom: Zoom level.
(Minimum value: 6, Maximum value: 19)
        height: Height of the map.
(Minimum value: 1, Maximum value: 999)
        datum: Geodetic system of latitude and longitude.
(wgs84: World Geodetic System (default), tokyo: Old Japan Geodetic System)
        coord_unit: The unit of latitude and longitude included in the output data.
(degree: decimal system of degrees (default), millisec: milliseconds)
        
    """
    url = f"https://navitime-maps.p.rapidapi.com/map_image"
    querystring = {'width': width, 'center': center, 'zoom': zoom, 'height': height, }
    if datum:
        querystring['datum'] = datum
    if coord_unit:
        querystring['coord_unit'] = coord_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navitime-maps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

