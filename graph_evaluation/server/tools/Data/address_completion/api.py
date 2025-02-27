import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_completion(text: str, limit: str='1', lon: str=None, lat: str=None, lang: str='en', countrycodes: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Address Autocomplete API allows developers to build location-based services like autocomplete fields. When a user enters a part of an address in the search box, the API provides suggestions for matching locations."
    text: Free-form address, place, region or area name
        limit: Maximum number of results
        lon: Location bias longitude (requires \"lat\")
        lat: Location bias latitude (requires \"lon\")
        lang: Preferable results language code (en, de, it, fr)
        countrycodes: Limit search to a comma-separated list of countries (ISO country codes)
        
    """
    url = f"https://address-completion.p.rapidapi.com/v1/geocode/autocomplete"
    querystring = {'text': text, }
    if limit:
        querystring['limit'] = limit
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if lang:
        querystring['lang'] = lang
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-completion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

