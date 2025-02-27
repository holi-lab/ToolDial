import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1(domainname: str, outputformat: str='JSON', mode: str='DNS_ONLY', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Domain Availability API helps you check whether a domain name is available for registration quickly and accurately."
    domainname: the domain for which domain info is requested
        outputformat: XML | JSON (defaults to JSON)
        mode: The default mode is the fastest, the **DNS_AND_WHOIS** mode is slower but more accurate.
Acceptable values: **DNS_AND_WHOIS | DNS_ONLY**
Default: **DNS_ONLY**
        
    """
    url = f"https://whoisapi-domain-availability-v1.p.rapidapi.com/api/v1"
    querystring = {'domainname': domainname, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi-domain-availability-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

