import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def open_proxies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of 200-400 working open proxies, updated every 15 minutes. Because they can go up & down, these IPs may not be functional when you retrieve them. For higher reliability, consider a [paid proxy service](https://proxymesh.com/)."
    
    """
    url = f"https://open-proxies.p.rapidapi.com/proxies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_open_proxies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of 200-400 working open proxies, updated every hour. Because they can go up & down, these IPs may not be functional when you retrieve them. For higher reliability, try the **Open Proxies** endpoint."
    
    """
    url = f"https://open-proxies.p.rapidapi.com/hourly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_open_proxies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of 200-400 open proxies, updated once per day around 00:00 UTC.  Because they can go up & down, these IPs may not be functional when you retrieve them. For higher reliability, try the **Hourly** or **Open Proxies** endpoints."
    
    """
    url = f"https://open-proxies.p.rapidapi.com/daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

