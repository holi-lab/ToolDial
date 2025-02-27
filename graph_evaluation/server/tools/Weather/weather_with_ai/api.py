import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_weather(version: str, lng: int, lat: int, unit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return weather in current, hourly and daily info."
    lng: Geographical coordinates of the location (longitude)
        lat: Geographical coordinates of the location (latitude)
        unit: Unit of temperature in Fahrenheit or Celsius.
(Kelvin is used by default if nothing selected)
        
    """
    url = f"https://weather-with-ai.p.rapidapi.com/get_weather/{version}"
    querystring = {'lng': lng, 'lat': lat, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-with-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

