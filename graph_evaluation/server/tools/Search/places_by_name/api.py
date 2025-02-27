import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def place_by_name(text: str, limit: str='1', type: str=None, countrycodes: str=None, lat: str='40.74', lang: str='en', lon: str='-73.98', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse, validate, and locate an address or place"
    text: Free-form address, place, region or area name
        limit: Maximum number of results
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        countrycodes: Limit search to a comma-separated list of countries (ISO country codes)
        lat: Location bias latitude (requires "lon")
        lang: Preferable results language code (en, de, it, fr)
        lon: Location bias longitude (requires "lat")
        
    """
    url = f"https://places-by-name.p.rapidapi.com/v1/geocode/search"
    querystring = {'text': text, }
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if lat:
        querystring['lat'] = lat
    if lang:
        querystring['lang'] = lang
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-by-name.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

