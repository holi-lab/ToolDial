import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(lat: str, lon: str, lang: str='en', type: str=None, limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide latitude and longitude coordinates in the "lat" and "lon" parameters to get the corresponding address back."
    lat: Latitude
        lon: Longitude
        lang: Preferable results language code (en, de, it, fr)
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        limit: Maximum number of results
        
    """
    url = f"https://location-to-address.p.rapidapi.com/v1/geocode/reverse"
    querystring = {'lat': lat, 'lon': lon, }
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-to-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

