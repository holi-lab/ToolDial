import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_dns_records(domain: str, record_type: str='a,mx', response_type_seperator: str=',', response_type: str='target', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a domains DNS records"
    record_type: Can be a comma seperated string with any of the following options:
a
ns
mx
txt
cname

To request multiple record types sample:
a,mx,cname
        response_type_seperator: When response_type is set to target, you can set the response_type_seperator here. Default is \\\\\\\\n (newline) but you can do comma or pipe delimited if you want.
        response_type: If response_type = target, will respond with either ip, txt or target of the requested dns record type. Useful when used in conjunction with response_type_seperator
        
    """
    url = f"https://vibrant-dns.p.rapidapi.com/dns/get"
    querystring = {'domain': domain, }
    if record_type:
        querystring['record_type'] = record_type
    if response_type_seperator:
        querystring['response_type_seperator'] = response_type_seperator
    if response_type:
        querystring['response_type'] = response_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vibrant-dns.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

