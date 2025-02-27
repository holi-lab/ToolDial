import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whoisserver_dnsservice(type: str, domainname: str, outputformat: str='JSON', callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DNS Lookup API"
    type: DNS type: A, NS, SOA, MX, etc. You can specify multiple comma-separated values, e.g., A,SOA,TXT; all records can be retrieved with type=_all.
Acceptable values: [A, NS, SOA, MX, etc.](https://dns-lookup.whoisxmlapi.com/api/documentation/making-requests#Supported-DNS-Types)
        domainname: The target domain name.
        outputformat: Response output format.
Acceptable values: JSON | XML
Default: XML
        callback: A javascript function used when outputFormat is JSON; this is an implementation known as JSONP which invokes the callback on the returned response.
        
    """
    url = f"https://whoisapi-dns-lookup-v1.p.rapidapi.com/whoisserver/DNSService"
    querystring = {'type': type, 'domainname': domainname, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi-dns-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

