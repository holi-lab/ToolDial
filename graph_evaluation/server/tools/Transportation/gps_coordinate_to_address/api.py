import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(lon: str, lat: str, type: str=None, lang: str='en', limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup an address and place information by coordinate or GPS location"
    lon: Longitude
        lat: Latitude
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        lang: Preferable results language code (en, de, it, fr)
        limit: Maximum number of results
        
    """
    url = f"https://gps-coordinate-to-address.p.rapidapi.com/v1/geocode/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gps-coordinate-to-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

