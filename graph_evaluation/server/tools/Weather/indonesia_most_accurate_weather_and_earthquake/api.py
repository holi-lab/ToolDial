import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_forecast_for_indonesian_cities_bmkg(province: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather forecast for Indonesian cities BMKG"
    
    """
    url = f"https://indonesia-most-accurate-weather-and-earthquake.p.rapidapi.com/weather/{province}/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-most-accurate-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_earth_quake_happened_on_indonesia_bmkg(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest earth quake happened on Indonesia"
    
    """
    url = f"https://indonesia-most-accurate-weather-and-earthquake.p.rapidapi.com/quake"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-most-accurate-weather-and-earthquake.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

