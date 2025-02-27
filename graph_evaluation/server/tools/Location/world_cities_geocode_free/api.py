import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocode_reverse(lon: int, lat: int, limit: int=3, radius_unit: str='km', radius: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of current and all nearby cities from a latitude and longitude of the particular location."
    
    """
    url = f"https://world-cities-geocode-free.p.rapidapi.com/geocode/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if limit:
        querystring['limit'] = limit
    if radius_unit:
        querystring['radius_unit'] = radius_unit
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities-geocode-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocode(city: str, limit: str='3', country: str='IN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search city geocode by partial name."
    
    """
    url = f"https://world-cities-geocode-free.p.rapidapi.com/geocode"
    querystring = {'city': city, }
    if limit:
        querystring['limit'] = limit
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities-geocode-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

