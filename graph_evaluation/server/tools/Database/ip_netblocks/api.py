import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_netblock_v2(asn: str=None, limit: str=None, org: str=None, outputformat: str=None, mask: str=None, ip: str='8.8.8.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exhaustive information on IP Range (v2)"
    asn: Get ranges by ASN (Autonomous System Number).
        limit: Max count of returned records. Acceptable values: 1 - 1000. Default: 100
        org: Find IP Netblocks which have the specified search terms in their Netblock (netname, description, remarks), or Organisation (org.org, org.name, org.email, org.address) fields
        outputformat: Response output format. Acceptable values: XML or JSON. Defaults to JSON.
        mask: Get ranges by CIDR. Acceptable values: 0 - 128 (0 - 32 for IPv4). Default: 128
        ip: Get ranges by IPv4/IPv6 address or by CIDR depending on input
        
    """
    url = f"https://ip-netblocks.p.rapidapi.com/api/v2"
    querystring = {}
    if asn:
        querystring['asn'] = asn
    if limit:
        querystring['limit'] = limit
    if org:
        querystring['org'] = org
    if outputformat:
        querystring['outputFormat'] = outputformat
    if mask:
        querystring['mask'] = mask
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-netblocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

