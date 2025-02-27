import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip(content_type: str, ip: str='237.140.128.110', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1、Lookup city, country, latitude and longitude of provided IP address.
		2、Your ip provided by query parameter will be used first, but if you do not pass ip param, we'll try best to analyze your ip from the request. 
		3、All Responses are in json format."
    ip: - Your ip provided by query parameter will be used first.
- If not, we'll try best to analyze your ip from the request you made.
        
    """
    url = f"https://fast-geo-ip.p.rapidapi.com/ip"
    querystring = {'Content-Type': content_type, }
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-geo-ip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

