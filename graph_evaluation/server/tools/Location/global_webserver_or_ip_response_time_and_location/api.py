import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geo_ping_global_ip_lookup(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Connects to 12 global servers and determines the local ip at that location for a given domain name or ip address, reports back location and response time of servers."
    
    """
    url = f"https://global-webserver-or-ip-response-time-and-location.p.rapidapi.com/geoping.php"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-webserver-or-ip-response-time-and-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

