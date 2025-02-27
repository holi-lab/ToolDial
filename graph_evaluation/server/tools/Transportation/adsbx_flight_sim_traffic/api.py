import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def livetraffic(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all aircraft within 25nm radius of specified Lat/Lon"
    lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180) Hint: US is negative longitude!
        
    """
    url = f"https://adsbx-flight-sim-traffic.p.rapidapi.com/api/aircraft/json/lat/{lat}/lon/{lon}/dist/25/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "adsbx-flight-sim-traffic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

