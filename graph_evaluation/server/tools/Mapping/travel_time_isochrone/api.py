import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reachability_area_isoline(type: str, lat: str, range: str, mode: str, lon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate isoline that defines reachability or service area (by travel time or distance)"
    type: Calculate by "time" or "distance"
        lat: Latitude of the starting location
        range: Maximum duration (seconds) or distance (meters)
        mode: Travel mode - drive, truck, bicycle, walk or transit
        lon: Longitude of the starting location
        
    """
    url = f"https://travel-time-isochrone.p.rapidapi.com/v1/isoline"
    querystring = {'type': type, 'lat': lat, 'range': range, 'mode': mode, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-time-isochrone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

