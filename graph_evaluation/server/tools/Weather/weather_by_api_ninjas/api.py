import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_weather(country: str=None, lon: str=None, zip: int=None, state: str=None, city: str='Seattle', lat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Weather API endpoint."
    country: Country name.
        lon: Longitude of desired location. If used, **lat** parameter must also be supplied.
        zip: 5-digit Zip code (United States only).
        state: US state (United States only).
        city: City name.
        lat: Latitude of desired location. If used, **lon** parameter must also be supplied.
        
    """
    url = f"https://weather-by-api-ninjas.p.rapidapi.com/v1/weather"
    querystring = {}
    if country:
        querystring['country'] = country
    if lon:
        querystring['lon'] = lon
    if zip:
        querystring['zip'] = zip
    if state:
        querystring['state'] = state
    if city:
        querystring['city'] = city
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

