import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def iplookup(ipaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific IP address, and whether it poses a threat to your network with a "Confidence of Abuse" score and the number of times it has been reported."
    
    """
    url = f"https://ip-threat-lookup.p.rapidapi.com/api/v2/check"
    querystring = {'ipAddress': ipaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-threat-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

