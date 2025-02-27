import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reversegeocode(location: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain address for location"
    location: The location for which you wish to obtain the human-readable address
        language: The language in which to return results
        
    """
    url = f"https://trueway-geocoding.p.rapidapi.com/ReverseGeocode"
    querystring = {'location': location, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocode(address: str, language: str='en', country: str=None, bounds: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain geocoordinates for address"
    address: The address that you want to geocode
        language: The language in which to return results
        country: The country code
        bounds: The bounding box
        
    """
    url = f"https://trueway-geocoding.p.rapidapi.com/Geocode"
    querystring = {'address': address, }
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if bounds:
        querystring['bounds'] = bounds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

