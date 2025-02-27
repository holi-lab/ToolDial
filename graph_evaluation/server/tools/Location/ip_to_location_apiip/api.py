import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_ip_data(callback: str='FUNCTION_NAME', ip: str='67.250.186.196', language: str='es', fields: str='city,capital', output: str='xml', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detailed information on our website: https://apiip.net/"
    callback: Specify a JSONP callback function name according to the [JSONP Callbacks](https://apiip.net/documentation) section.
        ip: An IPv4 or IPv6 address of your choice. or a comma-separated list of IPv4 or IPv6 addresses of your choice. (Limit: 50 values)
        language: Set to a 2-letter language code according to the [Specify Response Language](https://apiip.net/documentation) section.
        fields: Specify API response field(s) according to the [Specify Response Fields](https://apiip.net/documentation) section.
        output: Set to json or xml to choose between output formats.
        
    """
    url = f"https://ip-to-location-apiip.p.rapidapi.com/check"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if ip:
        querystring['ip'] = ip
    if language:
        querystring['language'] = language
    if fields:
        querystring['fields'] = fields
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-to-location-apiip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

