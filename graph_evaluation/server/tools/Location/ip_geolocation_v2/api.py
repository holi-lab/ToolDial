import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_address_geolocation_api(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The IP geolocation API is a web service used to identify internet users’ location information based on their IP address. The API provides highly accurate and detailed location data such as postal codes, city names, country names and much more."
    ip: IP address to lookup
        
    """
    url = f"https://bigdatacloud-ip-geolocation-v1-1.p.rapidapi.com/data/ip-geolocation"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigdatacloud-ip-geolocation-v1-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

