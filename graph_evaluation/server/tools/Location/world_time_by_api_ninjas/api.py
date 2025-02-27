import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_worldtime(timezone: str=None, lon: str=None, lat: str=None, state: str=None, country: str=None, city: str='London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas World Time API endpoint. Returns the current date and time by city/state/country, location coordinates (latitude/longitude), or timezone.
		
		One of the following parameter combinations must be set:
		lat + lon,
		city (state and country optional),
		timezone"
    timezone: Timezone of desired location (e.g. Europe/London).
        lon: Longitude of desired location.
        lat: Latitude of desired location.
        state: US state name or 2-letter abbreviation (for United States cities only).
        country: Country name or 2-letter country code.
        city: City name.
        
    """
    url = f"https://world-time-by-api-ninjas.p.rapidapi.com/v1/worldtime"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if state:
        querystring['state'] = state
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-time-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

