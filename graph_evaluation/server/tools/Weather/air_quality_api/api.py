import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def air_quality_measurements(country: str, city: str, lon: str='4.897070', lat: str='52.377956', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives current air quality measurements per hour for a given city!"
    
    """
    url = f"https://air-quality-api.p.rapidapi.com/air-quality/measurements"
    querystring = {'country': country, 'city': city, }
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def air_quality_forecasts(country: str, city: str, lat: str='52.377956', lon: str='4.897070', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives hourly air quality forecast for a given city!"
    
    """
    url = f"https://air-quality-api.p.rapidapi.com/air-quality/forecasts"
    querystring = {'country': country, 'city': city, }
    if lat:
        querystring['lat'] = lat
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_air_quality(country: str, city: str, lat: str='52.377956', lon: str='4.897070', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives back data of current air quality in a specified city!"
    
    """
    url = f"https://air-quality-api.p.rapidapi.com/air-quality"
    querystring = {'country': country, 'city': city, }
    if lat:
        querystring['lat'] = lat
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

