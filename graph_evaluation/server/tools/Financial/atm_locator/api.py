import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_atm_locations(longitude: int, latitude: int, radius: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search ATMs and their locations worldwide using latitude, longitude and an optional radius in kilometers"
    
    """
    url = f"https://atm-locator1.p.rapidapi.com/search"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "atm-locator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

