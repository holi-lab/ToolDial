import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crimea_check_api(longitude: str='34.4997', latitude: str='45.3453', street_address: str='Shmidta St, Simferopol', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks if an Latitude/Longitude or Street Address is within Crimea (the Crimean Peninsula)"
    longitude: The Longitude of the location to check.  This parameter is optional if the Street Address is set, but REQUIRED if Latitude is set.
        latitude: The Latitude of the location to check.  This parameter is optional if the Street Address is set, but REQUIRED if Longitude is set.
        street_address: The Street Address of the location to check.  This parameter is optional if latitude and longitude are set.
        
    """
    url = f"https://crimea-check.p.rapidapi.com/"
    querystring = {}
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if street_address:
        querystring['street_address'] = street_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crimea-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

