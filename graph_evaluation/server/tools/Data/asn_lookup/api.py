import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_information(ip: str=None, asn: str='AS32934', cidr: str=None, orgname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information based on AS number, CIDR (IPV4 & IPV6) or organization name."
    
    """
    url = f"https://asn-lookup.p.rapidapi.com/api"
    querystring = {}
    if ip:
        querystring['ip'] = ip
    if asn:
        querystring['asn'] = asn
    if cidr:
        querystring['cidr'] = cidr
    if orgname:
        querystring['orgname'] = orgname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asn-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

