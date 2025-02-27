import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_timezone(state: str=None, lon: str=None, city: str='London', lat: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Timezone API endpoint. Returns the timezone name of the specified input location.
		
		Either both (**lat** and **lon**) or (**city**/**state**/**country**) parameters must be set."
    state: US state name or 2-letter abbreviation (for United States cities only).
        lon: longitude of desired location.
        city: city name.
        lat: latitude of desired location.
        country: country name or 2-letter country code.
        
    """
    url = f"https://timezone-by-api-ninjas.p.rapidapi.com/v1/timezone"
    querystring = {}
    if state:
        querystring['state'] = state
    if lon:
        querystring['lon'] = lon
    if city:
        querystring['city'] = city
    if lat:
        querystring['lat'] = lat
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timezone-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

