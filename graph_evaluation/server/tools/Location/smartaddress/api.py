import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocodeencode(lng: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encode any GPS location (latitude, longitude) value into a more human readable address format"
    lng: Decimal value of Longitude of location input as string
        lat: Decimal value of Latitude of location input as string
        
    """
    url = f"https://smartaddress.p.rapidapi.com/api/geocode/v1/encode"
    querystring = {'lng': lng, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smartaddress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocodedecode(geocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decode GeoCode to GPS location (lat/lng value)"
    geocode: GeoCode value to decode to GPS lat/lng coordinate
        
    """
    url = f"https://smartaddress.p.rapidapi.com/api/geocode/v1/decode"
    querystring = {'geocode': geocode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smartaddress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

