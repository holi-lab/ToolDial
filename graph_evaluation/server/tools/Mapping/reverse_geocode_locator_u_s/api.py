import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup_coordinates(lat: int, long: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert US latitude and longitude coordinates into local city information"
    
    """
    url = f"https://reverse-geocode-locator-u-s.p.rapidapi.com/v1/geocode"
    querystring = {'lat': lat, 'long': long, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-geocode-locator-u-s.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

