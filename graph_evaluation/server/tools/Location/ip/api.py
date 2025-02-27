import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip(type: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IP address in plain text or Json format. Returns the visitor IP address , useful for shell scripts or to find the external Internet routable address."
    
    """
    url = f"https://ip9.p.rapidapi.com/api/ip"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location(ip_address: str='67.230.162.132', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "offers a JSON IP and REST API allowing to get a visitor IP address and to query location information from any IP address"
    
    """
    url = f"https://ip9.p.rapidapi.com/api/location"
    querystring = {}
    if ip_address:
        querystring['ip_address'] = ip_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

