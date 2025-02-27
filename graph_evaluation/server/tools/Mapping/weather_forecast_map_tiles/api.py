import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_weather_tile(y: int, z: int, x: int, time: int=30, unit: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a transparent weather tile according to the [Web Tile Standard(https://en.wikipedia.org/wiki/Tiled_web_map)."
    y: The y value of the requested tile
        z: The z value of the requested tile
        x: The x value of the requested tile
        time: The local time at which the weather should be shown. This is measured in hours since today 00:00. A time value of 10 would be today at 10:00 (even if this already passed). A time value of 40 would be tomorrow at 16:00 local time. Forecast tiles can be requested for 7 days. This results in a max time value of 168. Defaults to 'current'
        unit: The unit in which temperature and precipitation should be shown. Can be 'none', for no temperature or precipitation labels, 'metric' for temperature in °C and precipitation in mm or 'imperial' for temperature in °F and precipitation in inches. Defaults to 'metric'
        
    """
    url = f"https://weather-forecast-map-tiles.p.rapidapi.com/tile/{z}/{x}/{y}"
    querystring = {}
    if time:
        querystring['time'] = time
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

