import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(lon: str, lat: str, type: str=None, limit: str='1', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With our Reverse Geocoding API, you can get an address by its Lon/Lat coordinates. This makes it easy to define a user's location by GPS coordinates or find a building address on a map, for example, when a user clicks."
    lon: Longitude
        lat: Latitude
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        limit: Maximum number of results
        lang: Preferable results language code (en, de, it, fr)
        
    """
    url = f"https://address-details-by-location.p.rapidapi.com/v1/geocode/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-details-by-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

