import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def country_location_api(country: str='canada', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Country Location API is a RESTful API that allows developers to retrieve location data for any country in the world. Using a GET request with a country parameter, the API retrieves information about the specified country, such as its name, capital city, region, subregion, population, languages, and currencies."
    
    """
    url = f"https://country-location-api.p.rapidapi.com/location"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-location-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

