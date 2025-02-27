import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def as_info(asn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed information about an Autonomous System (AS) by AS number. The information includes registration, IPv4 address space announcements and ranking."
    asn: AS number to lookup, e.g. AS1234, ASN1234 or 1234
        
    """
    url = f"https://bigdatacloud-ip-to-asn-lookup-v1.p.rapidapi.com/data/asn-info"
    querystring = {'asn': asn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigdatacloud-ip-to-asn-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_to_network_and_as_lookup(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns detailed information about the active network a specific IP address belongs to. Including Autonomous Systems (AS) that announce and serve that network."
    ip: IPv4 address to lookup
        
    """
    url = f"https://bigdatacloud-ip-to-asn-lookup-v1.p.rapidapi.com/data/network-by-ip"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bigdatacloud-ip-to-asn-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

