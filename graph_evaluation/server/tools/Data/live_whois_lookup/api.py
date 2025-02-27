import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def live_whois_lookup(domainname: str, whois: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Complete Whois Detail With Domain Name"
    domainname: search with domain name
        format: available formats xml,json
        
    """
    url = f"https://live-whois-lookup.p.rapidapi.com/whois"
    querystring = {'domainName': domainname, 'whois': whois, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-whois-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_whois_lookup(domainname: str, whois: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Complete Whois Detail With Domain Name"
    domainname: search with domain name
        format: available formats xml,json
        
    """
    url = f"https://live-whois-lookup.p.rapidapi.com/whois"
    querystring = {'domainName': domainname, 'whois': whois, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-whois-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

