import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dns_request(type: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DNS Request
		
		Type can be one of:
		'A', 'AAAA', 'AFSDB', 'APL', 'CAA', 'CDNSKEY', 'CDS', 'CERT', 'CNAME', 'CSYNC', 'DHCID', 'DLV', 'DNAME', 'DNSKEY', 'DS', 'EUI48', 'EUI64', 'HINFO', 'HIP', 'HTTPS', 'IPSECKEY', 'KEY', 'KX', 'LOC', 'MX', 'NAPTR', 'NS', 'NSEC', 'NSEC3', 'NSEC3PARAM', 'OPENPGPKEY', 'PTR', 'RP', 'SMIMEA', 'SOA', 'SPF', 'SRV', 'SSHFP', 'SVCB', 'TA', 'TKEY', 'TLSA', 'TXT', 'URI', 'ZONEMD'
		
		IDN domains should be converted before requesting.
		
		Use **xn--fuball-cta.de** instead of **fußball.de**"
    
    """
    url = f"https://dns-records-lookup.p.rapidapi.com/"
    querystring = {'type': type, 'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dns-records-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

