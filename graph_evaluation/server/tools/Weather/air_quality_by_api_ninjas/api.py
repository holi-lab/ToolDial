import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_airquality(zip: str=None, country: str=None, city: str='Seattle', lon: int=None, state: str=None, lat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Air Quality API endpoint."
    country: Country name.
        city: City name.
        lon: Longitude of desired location. If used, the **lat** parameter must also be supplied.
        state: US state (United States only).
        lat: Latitude of desired location. If used, the **lon** parameter must also be supplied.
        
    """
    url = f"https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if lon:
        querystring['lon'] = lon
    if state:
        querystring['state'] = state
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

