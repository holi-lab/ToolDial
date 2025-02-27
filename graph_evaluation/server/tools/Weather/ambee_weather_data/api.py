import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_coordinates(lng: int, lat: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ambee’s Weather API enables weather forecasting by providing real-time weather data. Test an API call. Get accurate & actionable proprietary data insights."
    
    """
    url = f"https://ambee-weather-data.p.rapidapi.com/weather/by-lat-lng"
    querystring = {'lng': lng, 'lat': lat, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-weather-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

