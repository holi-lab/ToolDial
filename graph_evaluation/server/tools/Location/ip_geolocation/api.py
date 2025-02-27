import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_geo_location_information(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns full geo location information of the IP passed as argument: Continent, Country, Region, City, Postal code, Latitude and Longitude, Autonomous System and Reverse Hostname."
    
    """
    url = f"https://moocher-io-ip-geolocation-v1.p.rapidapi.com/{ip}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moocher-io-ip-geolocation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

