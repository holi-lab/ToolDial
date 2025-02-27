import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reversegeocode(lat: int, lon: int, format: str="'[SN[, ] - [23456789ab[, ]'", mode: str='text', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a text address from a latitude/longitude location"
    format: This is an advanced feature to get a custom formatted text.
Please read the available documentation at www.feroeg.com/api.html
        mode: 
The response is not in JSON, but as a single text string.
NOTE:  at the moment mode=text is the only option available. If omitted the response will be in JSON
        lang: This is the preferred language output. Use two char nation codes, as described in ISO 3166-1 alpha-2 specification
        
    """
    url = f"https://feroeg-reverse-geocoding.p.rapidapi.com/address"
    querystring = {'lat': lat, 'lon': lon, }
    if format:
        querystring['format'] = format
    if mode:
        querystring['mode'] = mode
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feroeg-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

