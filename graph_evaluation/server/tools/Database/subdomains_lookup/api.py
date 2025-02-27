import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1(domainname: str, outputformat: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subdomain Lookup API"
    domainname: The target domain name.
        outputformat: Response output format (JSON | XML).
Default: JSON
        
    """
    url = f"https://subdomains-lookup.p.rapidapi.com/api/v1"
    querystring = {'domainName': domainname, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subdomains-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

