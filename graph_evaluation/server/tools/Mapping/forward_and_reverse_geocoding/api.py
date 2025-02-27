import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forward_geocoding(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows to get latitude and longitude by address data inserted in a query as a string . Response of this API looks like Google Maps API compact response ."
    
    """
    url = f"https://forward-and-reverse-geocoding.p.rapidapi.com/geocode"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-and-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_geocoding(latlng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows to get address by latitude and longitude inserted in a query and separated by comma. Response of this API looks like Google Maps API compact response ."
    
    """
    url = f"https://forward-and-reverse-geocoding.p.rapidapi.com/reverse-geocode"
    querystring = {'latlng': latlng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-and-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

