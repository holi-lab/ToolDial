import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nearby_superchargers(lng: int, lat: int, radius: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby superchargers up to 50km away."
    radius: Kilometers. Default = 25. Max = 50.
        
    """
    url = f"https://nearby-tesla-superchargers.p.rapidapi.com/chargers"
    querystring = {'lng': lng, 'lat': lat, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-tesla-superchargers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

