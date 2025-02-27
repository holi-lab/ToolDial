import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip2proxy_api(ip: str, license: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Proxy Detection API"
    ip: IP address to query.
        license: API license key.
        
    """
    url = f"https://fraudlabs-ip2proxy-v1.p.rapidapi.com/ip2proxywebservice.asmx"
    querystring = {'IP': ip, 'LICENSE': license, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-ip2proxy-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

