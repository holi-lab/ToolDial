import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_completion(text: str, type: str='city', lat: str='52.51', countrycodes: str='de', lon: str='13.38', limit: str='1', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Address autocomplete endpoint for address forms. Send a part of an address and get address suggestions + their Latitude/Longitude coordinates."
    text: Free-form address, place, region or area name
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        lat: Location bias latitude (requires \"lon\")
        countrycodes: Limit search to a comma-separated list of countries (ISO country codes)
        lon: Location bias longitude (requires \"lat\")
        limit: Maximum number of results
        lang: Preferable results language code (en, de, it, fr)
        
    """
    url = f"https://geocode-address-to-location.p.rapidapi.com/v1/geocode/autocomplete"
    querystring = {'text': text, }
    if type:
        querystring['type'] = type
    if lat:
        querystring['lat'] = lat
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if lon:
        querystring['lon'] = lon
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocode-address-to-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocoding(text: str, limit: str='1', lat: str='40.74', type: str=None, countrycodes: str=None, lon: str='-73.98', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a street address and get the corresponding latitude/longitude coordinates, address parts, and the formatted address."
    text: Free-form address, place, region or area name
        limit: Maximum number of results
        lat: Location bias latitude (requires \"lon\")
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        countrycodes: Limit search to a comma-separated list of countries (ISO country codes)
        lon: Location bias longitude (requires \"lat\")
        lang: Preferable results language code (en, de, it, fr)
        
    """
    url = f"https://geocode-address-to-location.p.rapidapi.com/v1/geocode/search"
    querystring = {'text': text, }
    if limit:
        querystring['limit'] = limit
    if lat:
        querystring['lat'] = lat
    if type:
        querystring['type'] = type
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if lon:
        querystring['lon'] = lon
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocode-address-to-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

