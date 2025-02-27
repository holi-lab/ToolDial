import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reachability_area_isoline(mode: str, lat: str, type: str, lon: str, range: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate isoline that defines reachability or service area (by travel time or distance)"
    mode: Travel mode - drive, truck, bicycle, walk or transit
        lat: Latitude of the starting location
        type: Calculate by "time" or "distance"
        lon: Longitude of the starting location
        range: Maximum duration (seconds) or distance (meters)
        
    """
    url = f"https://calculate-service-area.p.rapidapi.com/v1/isoline"
    querystring = {'mode': mode, 'lat': lat, 'type': type, 'lon': lon, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "calculate-service-area.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

