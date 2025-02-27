import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_locator(ip_address: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API supports .csv, .xml and .json as output formats."
    format: Specify the desired format as part of the request uri and the response will be formatted accordingly.
        
    """
    url = f"https://ip-geolocation-metadata.p.rapidapi.com/{format}/{ip_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

