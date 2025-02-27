import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_country_by_ip(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide an IP address, get back the two-letter ISO country code, the country name and whether or not the country is part of the European Union"
    ip: The IP address. Can be IPv4 or IPv6.
        
    """
    url = f"https://getcountrybyip.p.rapidapi.com/countrybyip"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getcountrybyip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

