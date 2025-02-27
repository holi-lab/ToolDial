import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_to_coordinates_latitude_longitude(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By using this geographic tool you can get the lat long coordinates from an address. Please type the address which would include the name of the city/town, state and street name to get more accurate lat long value. Also, the gps coordinates of the address will be calculated below."
    
    """
    url = f"https://address-from-to-latitude-longitude.p.rapidapi.com/geolocationapi"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-from-to-latitude-longitude.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coordinates_latitude_longitude_to_address(lng: int, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By using this geographic tool you can get the lat long coordinates from an address. Please type the address which would include the name of the city/town, state and street name to get more accurate lat long value. Also, the gps coordinates of the address will be calculated below."
    
    """
    url = f"https://address-from-to-latitude-longitude.p.rapidapi.com/geolocationapi"
    querystring = {'lng': lng, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-from-to-latitude-longitude.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

