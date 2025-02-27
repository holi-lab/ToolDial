import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_geocoding(city: str, country: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Geocoding API endpoint."
    city: City name.
        country: Country name, 2-letter ISO country code, or 3-letter ISO country code.
        state: US state (for United States cities only).
        
    """
    url = f"https://geocoding-by-api-ninjas.p.rapidapi.com/v1/geocoding"
    querystring = {'city': city, }
    if country:
        querystring['country'] = country
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_reversegeocoding(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Reverse Geocoding API endpoint."
    lat: Latitude coordinate.
        lon: Longitude coordinate.
        
    """
    url = f"https://geocoding-by-api-ninjas.p.rapidapi.com/v1/reversegeocoding"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

