import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocoding_for_an_address(benchmark: str, address: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a geocode for a specified address"
    benchmark: A numerical ID or name that references what version of the locator should be searched.
        address: A single line containing the full address to be searched
        format: json or html
        
    """
    url = f"https://eec19846-geocoder-us-census-bureau-v1.p.rapidapi.com/locations/onelineaddress"
    querystring = {'benchmark': benchmark, 'address': address, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eec19846-geocoder-us-census-bureau-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocoding_and_geolookup_for_an_address(benchmark: str, address: str, format: str, vintage: str='Current_Current', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a geocode and a geolookup for a specified address"
    benchmark: A numerical ID or name that references what version of the locator should be searched.
        address: Address in one line
        format: json or html
        vintage: A numerical ID or name that references what vintage of geography is desired for the geoLookup.
        
    """
    url = f"https://eec19846-geocoder-us-census-bureau-v1.p.rapidapi.com/geographies/onelineaddress"
    querystring = {'benchmark': benchmark, 'address': address, 'format': format, }
    if vintage:
        querystring['vintage'] = vintage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eec19846-geocoder-us-census-bureau-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

