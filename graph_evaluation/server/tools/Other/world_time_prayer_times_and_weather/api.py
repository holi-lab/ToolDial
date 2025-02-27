import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def return_prayer_timings(city: str, get_prayers: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Time, Date, Timezone, State, Country, Lat Long and Prayer Data for the given city name"
    city: City Name 
        get_prayers: To get Prayer data of the given city
        
    """
    url = f"https://world-time-prayer-times-and-weather.p.rapidapi.com/city/{city}"
    querystring = {}
    if get_prayers:
        querystring['get_prayers'] = get_prayers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-time-prayer-times-and-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_prayer_timings_weather(city: str, get_weather: bool=None, get_prayers: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Time, Date, Timezone, State, Country, Lat Long, Prayer Data and Weather for the given city name"
    city: City Name 
        get_weather: To get Weather of the given city
        get_prayers: To get Prayer data of the given city
        
    """
    url = f"https://world-time-prayer-times-and-weather.p.rapidapi.com/city/{city}"
    querystring = {}
    if get_weather:
        querystring['get_weather'] = get_weather
    if get_prayers:
        querystring['get_prayers'] = get_prayers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-time-prayer-times-and-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_and_date(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Time, Date, Timezone, State, Country, Lat Long for the given city name"
    city: City Name 
        
    """
    url = f"https://world-time-prayer-times-and-weather.p.rapidapi.com/city/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-time-prayer-times-and-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

