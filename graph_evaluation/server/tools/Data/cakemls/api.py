import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocode(location: str='32.3074643,-110.9027209', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a text address from a latitude/longitude pair
		
		Forward and reverse geocoding both use the same geocode/ endpoint, but forward geocoding requires the "address" parameter while reverse geocoding requires the "location" parameter.
		
		The "location" parameter is required
		Latitude,Longitude    -   Negative sign is allowed
		
		*A trailing slash in the URL is REQUIRED."
    
    """
    url = f"https://cakemls.p.rapidapi.com/api/geocode/"
    querystring = {}
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cakemls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forward_geocode(address: str='4240+E+Aquarius+Dr%2C+Tuscon%2C+AZ+85718', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request latitude/longitude coordinates from a text address (address parameter)
		
		Forward and reverse geocoding both use the same geocode/ endpoint, but forward geocoding requires the "address" parameter while reverse geocoding requires the "location" parameter.
		
		*A trailing slash in the URL is REQUIRED."
    
    """
    url = f"https://cakemls.p.rapidapi.com/api/geocode/"
    querystring = {}
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cakemls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mls(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MLS data request for an individual dwelling
		GET parameters must be url-encoded
		GET parameters may contain encoded spaces
		The "address" parameter is required
		
		*A trailing slash in the URL is REQUIRED."
    
    """
    url = f"https://cakemls.p.rapidapi.com/api/mls/"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cakemls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

