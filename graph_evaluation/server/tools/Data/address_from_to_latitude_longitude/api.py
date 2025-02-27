import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def coordinates_latitude_longitude_to_address(lng: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By using this geographic tool you can get an address from lat long coordinates."
    
    """
    url = f"https://address-from-to-latitude-longitude1.p.rapidapi.com/GetLatLng"
    querystring = {'Lng': lng, 'Lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-from-to-latitude-longitude1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

