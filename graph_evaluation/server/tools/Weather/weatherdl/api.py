import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_forecast(lon: str='-122.4194', lat: str='37.7749', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request as string of a location's coordinates (in lat and lon) and downloads the full weather forecast of that location as API response."
    
    """
    url = f"https://weatherdl.p.rapidapi.com/weather"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherdl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

