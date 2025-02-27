import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocoding(text: str, bias: str='proximity:41.2257145,52.971411', type: str=None, filter: str='countrycode:de,es,fr', limit: str='1', lang: str='en', lon: str='-73.98', lat: str='40.74', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse, validate, and locate an address or place"
    text: Free-form address, place, region or area name
        bias: refer places by country, boundary, circle, location:
* by circle - circle:lon,lat,radiusMeters, for example, bias=circle:-87.770231,41.878968,5000
* by rectangle - rect:lon1,lat1,lon2,lat2, for example , bias=rect:-89.097540,39.668983,-88.399274,40.383412
* by country - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case),  for example, bias=countrycode:de,es,fr
* by proximity - proximity:lon,lat, for example, bias=proximity:41.2257145,52.971411
        type: Desired result granularity  (country, state, city, postcode, street, or amenity)
        limit: Maximum number of results
        lang: Preferable results language code (en, de, it, fr)
        lon: Location bias longitude (requires \"lat\")
        lat: Location bias latitude (requires \"lon\")
        
    """
    url = f"https://verify-and-geocode-address.p.rapidapi.com/v1/geocode/search"
    querystring = {'text': text, }
    if bias:
        querystring['bias'] = bias
    if type:
        querystring['type'] = type
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "verify-and-geocode-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

