import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_weather_data(q: str, is_id: str='2172797', lat: str=None, mode: str='xml, html', units: str='"metric" or "imperial"', lon: str=None, callback: str='test', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this kind of requests you can get weather data in any location on the earth. The current weather data are updated online based on data from more than 40,000 weather stations."
    q: use this parameter when searching for a city. Do not use with other parameters
        
    """
    url = f"https://bowenapi.p.rapidapi.com/weather"
    querystring = {'q': q, }
    if is_id:
        querystring['id'] = is_id
    if lat:
        querystring['lat'] = lat
    if mode:
        querystring['mode'] = mode
    if units:
        querystring['units'] = units
    if lon:
        querystring['lon'] = lon
    if callback:
        querystring['callback'] = callback
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bowenapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

