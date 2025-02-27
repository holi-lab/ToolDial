import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_reputation(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch detailed information about the specified IPv4 or IPv6 address. Get the country by geolocation, ISP or hosting provider. You will also find out if this IP address has a history of abuse or malicious activity."
    
    """
    url = f"https://ip-reputation-geoip-and-detect-vpn.p.rapidapi.com/"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-reputation-geoip-and-detect-vpn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

