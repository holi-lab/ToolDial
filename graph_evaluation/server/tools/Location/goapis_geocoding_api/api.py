import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocding(key: str='karan', lat: str='37.7749', long: str='-122.4194', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geocoding is the process of converting a physical address into geographic coordinates, such as latitude and longitude. It allows for the identification of a specific location on the Earth's surface, making it possible to map and analyze data spatially"
    
    """
    url = f"https://goapis-geocoding-api.p.rapidapi.com/"
    querystring = {}
    if key:
        querystring['key'] = key
    if lat:
        querystring['lat'] = lat
    if long:
        querystring['long'] = long
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goapis-geocoding-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

