import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whois_lookup_v2(domainname: str, thinwhois: str='0', callback: str=None, preferfresh: str='0', checkproxydata: str='0', parse: str='0', registryrawtext: str=None, registrarrawtext: str=None, ipwhois: str='0', ip: str='0', da: str='0', outputformat: str='XML', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WHOIS lookup in JSON or XML, ver. 2"
    domainname: The name of the domain looked up.
        thinwhois: 1 results in returning whois data from the registry only, without fetching data from the registrar. Returned registry data are to be found in the WhoisRecord → registryData schema element. Accepted values are 0 and 1, the default is 0.
        callback: A JAVAscript function used when outputFormat is JSON; this is an implementation known as JSONP which invokes the callback on the returned response.
        preferfresh: Can be 0 or 1. 1 results in getting the latest Whois record even if it is incomplete. Defaults to 0.
        checkproxydata: 1 results in fetching proxy/whois guard data, if found, they are returned in the WhoisRecord → privateWhoisProxy schema element. Accepted values are 0 and 1, the default is 0.
        parse: This parameter enables the use of the API for parsing WHOIS raw text provided in the query.1 provides parsing for input WHOIS raw texts described at the parameters registryRawText and registrarRawText. Accepted values are 0 and 1, the default is 0.
        registryrawtext: The string containing the registry WHOIS raw text to be parsed; works only when the _parse parameter is equal to 1.
        registrarrawtext: A string containing the registrar WHOIS raw text to be parsed; works only when the _parse parameter is equal to 1.
        ipwhois: 1 results in returning the WHOIS record for the hosting IP if the WHOIS record for the domain is not supported. For unsupported TLDs, domain NameServers are returned if the ipWhois flag is activated, a WHOIS record for the hosting IP is added to the result.  Accepted values are 0 and 1, the default is 0.
        ip: Return IPs for the domain name. Accepted values are 0 and 1. Defaults to 0: no IPs are returned. 1 results in returning IPs.
        da: Perform domain availability check. Accepted values are 0, 1 and 2. Defaults to 0: no domain availability check is performed. When set to 1 the result contains a quick check on domain availability, 2 is slower but more accurate. Results are returned under WhoisRecord → domainAvailability (AVAILABLE | UNAVAILABLE | UNDETERMINED).
        outputformat: Response output format.  Acceptable values: XML or JSON. Defaults to XML.
        
    """
    url = f"https://whoisapi-whois-v2-v1.p.rapidapi.com/whoisserver/WhoisService"
    querystring = {'domainName': domainname, }
    if thinwhois:
        querystring['thinWhois'] = thinwhois
    if callback:
        querystring['callback'] = callback
    if preferfresh:
        querystring['preferfresh'] = preferfresh
    if checkproxydata:
        querystring['checkproxydata'] = checkproxydata
    if parse:
        querystring['_parse'] = parse
    if registryrawtext:
        querystring['registryrawtext'] = registryrawtext
    if registrarrawtext:
        querystring['registrarRawText'] = registrarrawtext
    if ipwhois:
        querystring['ipwhois'] = ipwhois
    if ip:
        querystring['ip'] = ip
    if da:
        querystring['da'] = da
    if outputformat:
        querystring['outputFormat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi-whois-v2-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

