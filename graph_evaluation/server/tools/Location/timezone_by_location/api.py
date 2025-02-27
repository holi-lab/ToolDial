import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def timezone(lat: int, lon: int, c: int=1, s: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a position into its timezone"
    lat: Latitude
        lon: Longitude
        c: Return compact JSON
        s: Return simple answer (ignores points with multiple timezones)
        
    """
    url = f"https://timezone-by-location.p.rapidapi.com/timezone"
    querystring = {'lat': lat, 'lon': lon, }
    if c:
        querystring['c'] = c
    if s:
        querystring['s'] = s
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timezone-by-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

