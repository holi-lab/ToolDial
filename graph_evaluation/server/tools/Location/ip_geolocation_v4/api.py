import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1(ipaddress: str, outputformat: str=None, domain: str=None, email: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Geolocation in XML or JSON, v1"
    ipaddress: IPv4 or IPv6 to search location by.
        outputformat: Response output format. Acceptable values: JSON | XML Default: JSON
        domain: Domain name to search location by
        email: Email address or domain name to search location by it's MX servers
        
    """
    url = f"https://whoisapi-ip-geolocation-v1.p.rapidapi.com/api/v1"
    querystring = {'ipAddress': ipaddress, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    if domain:
        querystring['domain'] = domain
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi-ip-geolocation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

