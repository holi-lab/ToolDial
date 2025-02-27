import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def addressautocomplete(text: str, bias: str='proximity:10.485306,48.852565', filter: str='countrycode:de,es,fr', lang: str='de', limit: str='2', type: str='street', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Address Autocomplete API is used to implement location autocomplete fields. In general, the API is called when a user presses a key in the address field to show address suggestions in a dropdown list."
    bias: Prefer places by country, boundary, circle, location:
* by circle - circle:lon,lat,radiusMeters, for example, bias=circle:-87.770231,41.878968,5000
* by rectangle - rect:lon1,lat1,lon2,lat2, for example , bias=rect:-89.097540,39.668983,-88.399274,40.383412
* by country - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case),  for example, bias=countrycode:de,es,fr
* by proximity - proximity:lon,lat, for example, bias=proximity:41.2257145,52.971411
        filter: Filter places by country, boundary, circle:
* by circle - circle:lon,lat,radiusMeters, for example, filter=circle:-87.770231,41.878968,5000
* by rectangle - rect:lon1,lat1,lon2,lat2, for example , filter=rect:-89.097540,39.668983,-88.399274,40.383412
* by country - comma-separated ISO 3166-1 Alpha-2 country codes (in lower case),  for example, filter=countrycode:de,es,fr
        type: Desired result granularity. Possible values: 'country', 'state', 'city', 'postcode', 'street', 'amenity'.
        
    """
    url = f"https://geoapify-address-autocomplete.p.rapidapi.com/v1/geocode/autocomplete"
    querystring = {'text': text, }
    if bias:
        querystring['bias'] = bias
    if filter:
        querystring['filter'] = filter
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapify-address-autocomplete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

