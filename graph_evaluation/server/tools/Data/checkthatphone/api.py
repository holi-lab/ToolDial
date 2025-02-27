import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def checkthatphone(phone: str, ip: str='172.95.90.55', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validation, carrier look-up, line type, portability status, geoIP and timezone, routing.
		Simply provide a United States phone number and an ip (optional) to get a result."
    
    """
    url = f"https://checkthatphone.p.rapidapi.com/checkthatphone"
    querystring = {'phone': phone, }
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkthatphone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

