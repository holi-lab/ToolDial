import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categories of establishments."
    
    """
    url = f"https://places-nearby-a-coordinates.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-nearby-a-coordinates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby(lon: int, lat: int, categories: str='catering.cafe', radius: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns nearby places sorted by distance from the origin coordinates in ascending order.
		
		Returns up to 60 places per request."
    lon: Longitude
        lat: Latitude
        categories: Default: catering.cafe
        radius: **Meters**
Default: 200
Maximum: 50000
        
    """
    url = f"https://places-nearby-a-coordinates.p.rapidapi.com/nearby"
    querystring = {'lon': lon, 'lat': lat, }
    if categories:
        querystring['categories'] = categories
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-nearby-a-coordinates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

