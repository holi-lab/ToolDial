import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def air_quality_history(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the past 24 hours of air quality observations for any point in the world given a lat/lon."
    lat: Latitude
        lon: Longitude
        
    """
    url = f"https://air-quality.p.rapidapi.com/history/airquality"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def air_quality_forecast(lat: int, lon: int, hours: int=72, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a 3 day (72 hour) air quality forecast for any point in the world given a lat/lon. "
    lat: Latitude
        lon: Longitude
        hours: Limits response forecast hours (default 72). 
        
    """
    url = f"https://air-quality.p.rapidapi.com/forecast/airquality"
    querystring = {'lat': lat, 'lon': lon, }
    if hours:
        querystring['hours'] = hours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_air_quality(lon: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves current air quality conditions for any location in the world, given a lat/lon."
    lon: Longitude
        lat: Latitude
        
    """
    url = f"https://air-quality.p.rapidapi.com/current/airquality"
    querystring = {'lon': lon, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

