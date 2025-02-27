import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getgeocodingdata(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latitude, longitude and other information"
    location: A string representing a location (an address or city)
        
    """
    url = f"https://geocoding-and-reverse-geocoding.p.rapidapi.com/geocode/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-and-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgeocodingdataforsearchmatches(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latitude, longitude for locations that may match the search string - useful for building search suggest or autocomplete functionality"
    
    """
    url = f"https://geocoding-and-reverse-geocoding.p.rapidapi.com/search/{text}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-and-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreversegeocodingdata(coordinates: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns location information"
    coordinates: Comma separated co-ordinates in the order of latitude,longitude
        
    """
    url = f"https://geocoding-and-reverse-geocoding.p.rapidapi.com/reverse/{coordinates}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-and-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

