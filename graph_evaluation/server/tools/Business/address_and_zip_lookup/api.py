import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocoding(text: str, bias: str='proximity:41.2257145,52.971411', filter: str='countrycode:de,es,fr', lat: str='40.74', limit: str='1', lon: str='-73.98', type: str='postcode', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse, validate, and locate an address or place"
    text: Free-form address, place, region or area name
        bias: Prefer places by country, boundary, circle, location:
* by circle - circle:lon,lat,radiusMeters, for example, bias=circle:-87.770231,41.878968,5000
* by rectangle - rect:lon1,lat1,lon2,lat2, for example , bias=rect:-89.097540,39.668983,-88.399274,40.383412
* by country - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case),  for example, bias=countrycode:de,es,fr
* by proximity - proximity:lon,lat, for example, bias=proximity:41.2257145,52.971411
        filter: Filter places by country, boundary, circle:
* by circle - circle:lon,lat,radiusMeters, for example, filter=circle:-87.770231,41.878968,5000
* by rectangle - rect:lon1,lat1,lon2,lat2, for example , filter=rect:-89.097540,39.668983,-88.399274,40.383412
* by country - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case),  for example, filter=countrycode:de,es,fr
        lat: Location bias latitude (requires \\\\\\\"lon\\\\\\\")
        limit: Maximum number of results
        lon: Location bias longitude (requires \\\\\\\"lat\\\\\\\")
        type: Desired result granularity. Possible values: 'country', 'state', 'city', 'postcode', 'street', 'amenity'.
        lang: Preferable results language code (en, de, it, fr)
        
    """
    url = f"https://address-and-zip-lookup.p.rapidapi.com/v1/geocode/search"
    querystring = {'text': text, }
    if bias:
        querystring['bias'] = bias
    if filter:
        querystring['filter'] = filter
    if lat:
        querystring['lat'] = lat
    if limit:
        querystring['limit'] = limit
    if lon:
        querystring['lon'] = lon
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-and-zip-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

