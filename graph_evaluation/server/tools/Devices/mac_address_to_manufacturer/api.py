import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def maclookup(mac_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter MAC Address (string) to find the manufacturer. MAC Formats Allowed: ('xx-xx-xx-xx-xx-xx', 'xx:xx:xx:xx:xx:xx', 'xxxx.xxxx.xxxx', 'xxxxxxxxxxxx')"
    mac_number: MAC Address Number to Find Manufacturer
        
    """
    url = f"https://mac-address-to-manufacturer.p.rapidapi.com/maclookup/{mac_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mac-address-to-manufacturer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

