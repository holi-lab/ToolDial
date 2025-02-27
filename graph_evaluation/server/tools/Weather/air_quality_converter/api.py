import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aqitable(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HTML full gradient reference table AQI / PM2.5"
    
    """
    url = f"https://air-quality-converter.p.rapidapi.com/AQItable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pm25tousaqi(pm25: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts PM2.5 exposure value in µg/m³ to its corresponding value on the US AQI scale (U.S. Air Quality Index, https://www.airnow.gov/aqi/aqi-basics/)
		
		With official description and color codes."
    
    """
    url = f"https://air-quality-converter.p.rapidapi.com/PM25toUSAQI"
    querystring = {'pm25': pm25, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usaqitopm25(aqi: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts US AQI (U.S. Air Quality Index, https://www.airnow.gov/aqi/aqi-basics/) value to its corresponding PM2.5 exposure value in µg/m³.
		
		With official description and color codes."
    
    """
    url = f"https://air-quality-converter.p.rapidapi.com/USAQItoPM25"
    querystring = {'aqi': aqi, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

