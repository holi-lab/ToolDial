import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_weather_by_lat_long(lat: str, long: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves current US weather information including air quality, sunrise, and sunset"
    
    """
    url = f"https://us-weather-by-lat-long.p.rapidapi.com/getweatherlatlong"
    querystring = {'lat': lat, 'long': long, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-weather-by-lat-long.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

