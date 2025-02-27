import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(lon: str, lat: str, limit: str='1', lang: str='fr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse Geocoding API returns a location with an address by Lon/Lat coordinates. It's widely used to define user location by GPS coordinates or find a building address on a map, for example, by user click."
    
    """
    url = f"https://geoapify-reverse-geocoding.p.rapidapi.com/v1/geocode/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapify-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

