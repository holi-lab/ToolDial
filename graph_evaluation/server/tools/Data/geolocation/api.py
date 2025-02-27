import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geolocation(ip: str='200.192.123.145', callback: str='callback12345', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns geolocation information for the given IP address. If no IP address is specified, then the current IP address is used."
    ip: IP Address to be looked upon. If this argument is omitted, the current user's IP address is used.
        callback: If this argument is specified, the geolocation API will return the results as JSONP. If this is not specified, a JSON response is sent.
        
    """
    url = f"https://geo.p.rapidapi.com/"
    querystring = {}
    if ip:
        querystring['ip'] = ip
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

