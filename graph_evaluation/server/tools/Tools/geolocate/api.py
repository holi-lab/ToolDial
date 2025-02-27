import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_coordinates(address: str='1600 Amphitheatre Parkway, Mountain View, CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET request to the "/geocode" endpoint with an "address" parameter containing the address or place you want to geocode"
    
    """
    url = f"https://geolocate4.p.rapidapi.com/geocode"
    querystring = {}
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geolocate4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

