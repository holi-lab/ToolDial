import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def batch_reverse_geocoding(coordinates: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows to to reverse geocode large data sets. You can reverse geocode a lot of latitudes and longitudes at once, as minimum - starting from one set of coordinates and ending with maximum 100 sets of latitudes and longitudes. Response of this API looks like Google Maps API compact response ."
    coordinates: An array of coordinates (latitude and longitude values specifying the location for which you wish to obtain the closest, human-readable address). Latitude and Longitude should be delimited by comma. They should be specified in an array and delimited by comma.
        
    """
    url = f"https://batch-geocoding.p.rapidapi.com/reverse-geocode-batch"
    querystring = {'coordinates': coordinates, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batch-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def batch_forward_geocoding(addresses: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows to to geocode large data sets. You can geocode a lot of addresses at once, as minimum - starting from one address and ending with maximum 100 addresses. Response of this API looks like Google Maps API compact response ."
    addresses: An array of street addresses that you want to geocode. Specify addresses in accordance with the format used by the national postal service of the country concerned. Additional address elements such as business names and unit, suite or floor numbers should be avoided. Street address elements should be delimited by spaces. Addresses should be specified in an array and delimited by comma.
        
    """
    url = f"https://batch-geocoding.p.rapidapi.com/geocode-batch"
    querystring = {'addresses': addresses, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batch-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

