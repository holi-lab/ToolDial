import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip(ip: str='8.8.8.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data for the provided IP"
    
    """
    url = f"https://ip-geo-location-and-ip-reputation.p.rapidapi.com/"
    querystring = {}
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geo-location-and-ip-reputation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def format(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Format to display data
		Can be "json" or "xml"
		By default will be select json"
    
    """
    url = f"https://ip-geo-location-and-ip-reputation.p.rapidapi.com/"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geo-location-and-ip-reputation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

