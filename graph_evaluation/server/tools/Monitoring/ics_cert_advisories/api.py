import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_advisories_by_vendor(vendorid: str, limit: int=2, info: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back Advisories by Vendor and provide timely information about current security issues, vulnerabilities, and exploits from CISA.gov"
    
    """
    url = f"https://ics-cert-advisories.p.rapidapi.com/advisories/{vendorid}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if info:
        querystring['info'] = info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ics-cert-advisories.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_advisories(info: bool=None, limit: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all  advisories and provide timely information about current security issues, vulnerabilities, and exploits from CISA.gov"
    
    """
    url = f"https://ics-cert-advisories.p.rapidapi.com/advisories"
    querystring = {}
    if info:
        querystring['info'] = info
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ics-cert-advisories.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

