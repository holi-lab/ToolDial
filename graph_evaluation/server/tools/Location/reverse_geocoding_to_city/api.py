import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding_api(latitude: int, longitude: int, localitylanguage: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts latitude and longitude into address."
    
    """
    url = f"https://reverse-geocoding-to-city.p.rapidapi.com/data/reverse-geocode"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if localitylanguage:
        querystring['localityLanguage'] = localitylanguage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-geocoding-to-city.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

