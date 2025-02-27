import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip2location_api(license: str, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Geolocation API"
    license: API license key
        ip: IP address to query
        
    """
    url = f"https://fraudlabs-ip2location-v1.p.rapidapi.com/ip2locationwebservice.asmx"
    querystring = {'license': license, 'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-ip2location-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

