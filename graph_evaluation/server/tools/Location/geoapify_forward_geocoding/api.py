import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocoding(text: str, limit: int=1, bias: str='proximity:41.2257145,52.971411', filter: str='countrycode:de,es,fr', type: str='street', lang: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geoapify provides Geocoding API that searches addresses worldwide. The API works via HTTP GET API. So it's cross-platform and can be used with most of the programming languages."
    bias: Prefer places by country, boundary, circle, location:
* By circle - circle:lon,lat,radiusMeters - bias=circle:-87.770231,41.878968,5000
* By rectangle - rect:lon1,lat1,lon2,lat2 - bias=rect:-89.097540,39.668983,-88.399274,40.383412
* By country - Comma-separated ISO 3166-1 Alpha-2 country codes (in lower case). Use 'auto' to detect the country by IP address. Use 'none' to skip (default value). - bias=countrycode:de,es,fr
* By location - proximity:lon,lat - bias=proximity:41.2257145,52.971411
        filter: Filter places by country, boundary, circle:
* By circle - circle:lon,lat,radiusMeters -  filter=circle:-87.770231,41.878968,5000
* By rectangle - rect:lon1,lat1,lon2,lat2 - 	filter=rect:-89.097540,39.668983,-88.399274,40.383412
* By country  - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case). Use 'auto' to detect the country by IP address. Use 'none' to skip (default value). - filter=countrycode:de,es,fr
        type: Location type. Possible values: 'country', 'state', 'city', 'postcode', 'street', 'amenity'.
        lang: Result language. 2-character ISO 639-1 language codes are supported.
        
    """
    url = f"https://geoapify-forward-geocoding.p.rapidapi.com/v1/geocode/search"
    querystring = {'text': text, }
    if limit:
        querystring['limit'] = limit
    if bias:
        querystring['bias'] = bias
    if filter:
        querystring['filter'] = filter
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapify-forward-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

