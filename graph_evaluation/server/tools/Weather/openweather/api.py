import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcurrentweather(appid: str, q: str, lang: str=None, units: str='standard', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test"
    q: City name
        lang: Parameter to get the output in your language. Translation is applied for the city name and description fields
        units: Units of measurement. For temperature in Fahrenheit imperial; For temperature in Celsius - metric; for temperature in Kelvin - standart
        
    """
    url = f"https://openweather43.p.rapidapi.com/weather"
    querystring = {'appid': appid, 'q': q, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openweather43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforecastweather(q: str, cnt: int=None, units: str='standard', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    q: City name
        cnt: A number of timestamps, which will be returned in the API response.
        units: Units of measurement. For temperature in Fahrenheit imperial; For temperature in Celsius - metric; for temperature in Kelvin - standart
        lang: Parameter to get the output in your language. Translation is applied for the city name and description fields
        
    """
    url = f"https://openweather43.p.rapidapi.com/forecast"
    querystring = {'q': q, }
    if cnt:
        querystring['cnt'] = cnt
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openweather43.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

