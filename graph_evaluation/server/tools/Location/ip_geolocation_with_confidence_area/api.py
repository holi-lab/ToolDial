import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_geolocation_with_confidence_area(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Geolocation with Confidence Area"
    ip: IP address to lookup in string or decimal format e.g. 8.8.8.8 or 134744072
        
    """
    url = f"https://bigdatacloud-ip-geolocation-with-confidence-area-v1.p.rapidapi.com/data/ip-geolocation-with-confidence"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigdatacloud-ip-geolocation-with-confidence-area-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

