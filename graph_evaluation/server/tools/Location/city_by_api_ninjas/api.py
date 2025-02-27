import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_city(max_lat: int=None, min_population: int=None, min_lon: int=None, country: str='US', max_population: int=None, min_lat: int=None, name: str='New York', max_lon: int=None, limit: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas City API endpoint."
    max_lat: Maximum latitude coordinate.
        min_population: Minimum city population.
        min_lon: Minimum longitude coordinate.
        country: Country filter. Must be an ISO-3166 alpha-2 code (e.g. **US**).
        max_population: Maximum city population.
        min_lat: Minimum latitude coordinate.
        name: Name of city.
        max_lon: Maximum longitude coordinate.
        limit: How many results to return. Must be between **1** and **30**. Default is **1**.
        
    """
    url = f"https://city-by-api-ninjas.p.rapidapi.com/v1/city"
    querystring = {}
    if max_lat:
        querystring['max_lat'] = max_lat
    if min_population:
        querystring['min_population'] = min_population
    if min_lon:
        querystring['min_lon'] = min_lon
    if country:
        querystring['country'] = country
    if max_population:
        querystring['max_population'] = max_population
    if min_lat:
        querystring['min_lat'] = min_lat
    if name:
        querystring['name'] = name
    if max_lon:
        querystring['max_lon'] = max_lon
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

