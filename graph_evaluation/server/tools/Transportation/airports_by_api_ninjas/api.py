import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_airports(offset: int=None, max_elevation: int=None, region: str=None, min_elevation: int=None, city: str=None, country: str=None, timezone: str=None, name: str='London Heathrow', icao: str=None, iata: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Airports API endpoint. At least one of the optional parameters must be provided."
    offset: number of results to offset for pagination.
        max_elevation: maximum airport elevation in feet.
        region: administrative region such as state or province within a country (e.g. California)
        min_elevation: minimum airport elevation in feet.
        city: airport city (e.g. London)
        country: airport country. Must be 2-character ISO-2 country code (e.g. GB)
        timezone: airport timezone (e.g. Europe/London)


        name: airport name. This parameter supports partial matching (e.g. Heathrow will match London Heathrow Airport)
        icao: International Civil Aviation Organization (ICAO) 4-character airport code.
        iata: International Air Transport Association (IATA) 3-character airport code.
        
    """
    url = f"https://airports-by-api-ninjas.p.rapidapi.com/v1/airports"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if max_elevation:
        querystring['max_elevation'] = max_elevation
    if region:
        querystring['region'] = region
    if min_elevation:
        querystring['min_elevation'] = min_elevation
    if city:
        querystring['city'] = city
    if country:
        querystring['country'] = country
    if timezone:
        querystring['timezone'] = timezone
    if name:
        querystring['name'] = name
    if icao:
        querystring['icao'] = icao
    if iata:
        querystring['iata'] = iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airports-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

