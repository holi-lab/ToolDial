import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def commonports(port: str='53', protocol: str='tcp', service: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get registered service names and transport protocols port numbers."
    
    """
    url = f"https://commonportnumbers.p.rapidapi.com/commonportnames"
    querystring = {}
    if port:
        querystring['port'] = port
    if protocol:
        querystring['protocol'] = protocol
    if service:
        querystring['service'] = service
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commonportnumbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

