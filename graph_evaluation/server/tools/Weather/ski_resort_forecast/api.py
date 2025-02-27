import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_5_day_forecast(resort: str, units: str='i', el: str='top', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 5 day forecast for a given resort name"
    units: Specify desired units. Accepted values are 'i' (imperial) and 'm' (metric). Using this parameter improves response time.
        el: Specify a part of the mountain. Accepted values are 'top', 'mid and 'bot'. Using this parameter improves response time.
        
    """
    url = f"https://ski-resort-forecast.p.rapidapi.com/{resort}/forecast"
    querystring = {}
    if units:
        querystring['units'] = units
    if el:
        querystring['el'] = el
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ski-resort-forecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_forecast(resort: str, el: str='top', units: str='i', c: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the hourly forecast for a given resort name"
    el: Specify a part of the mountain. Accepted values are 'top', 'mid', and 'bot'. Using this parameter improves response time.
        units: Specify desired units. Accepted values are 'i' (imperial) and 'm' (metric). Using this parameter improves response time.
        c: Limit result to current day only
        
    """
    url = f"https://ski-resort-forecast.p.rapidapi.com/{resort}/hourly"
    querystring = {}
    if el:
        querystring['el'] = el
    if units:
        querystring['units'] = units
    if c:
        querystring['c'] = c
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ski-resort-forecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_snow_conditions(resort: str, units: str='i', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current snow conditions for a given resort name"
    units: Specify desired units. Accepted values are 'i' (imperial) and 'm' (metric). Using this parameter improves response time.
        
    """
    url = f"https://ski-resort-forecast.p.rapidapi.com/{resort}/snowConditions"
    querystring = {}
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ski-resort-forecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

