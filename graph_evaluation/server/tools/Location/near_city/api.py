import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nearest_city(longitude: str, latitude: str, unit: str='miles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API end point provides you the easy way to convert geographic coordinates into city information with the distance difference between the city and given coordinates also gives the list of other nearest cities."
    
    """
    url = f"https://near-city.p.rapidapi.com/"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "near-city.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

