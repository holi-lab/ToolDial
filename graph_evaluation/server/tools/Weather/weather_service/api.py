import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_forecast_30_days(units: str, language: str, icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather provides local & long-range weather forecasts, weather reports"
    
    """
    url = f"https://weather-service.p.rapidapi.com/30day"
    querystring = {'units': units, 'language': language, 'icaoCode': icaocode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_forecast_10_days(units: str, language: str, icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather provides local & long-range weather forecasts, weather reports"
    
    """
    url = f"https://weather-service.p.rapidapi.com/10day"
    querystring = {'units': units, 'language': language, 'icaoCode': icaocode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_forecast_3_days(language: str, icaocode: str, units: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather provides local & long-range weather forecasts, weather reports"
    
    """
    url = f"https://weather-service.p.rapidapi.com/3day"
    querystring = {'language': language, 'icaoCode': icaocode, 'units': units, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_forecast_7_days(icaocode: str, units: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weather provides local & long-range weather forecasts, weather reports"
    
    """
    url = f"https://weather-service.p.rapidapi.com/7day"
    querystring = {'icaoCode': icaocode, 'units': units, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

