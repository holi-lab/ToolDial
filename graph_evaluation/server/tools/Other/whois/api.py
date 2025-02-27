import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whois_lookup_v1(domainname: str, ipwhois: str=None, preferfresh: str=None, ignorerawtexts: str=None, thinwhois: str=None, callback: str=None, registryrawtext: str=None, outputformat: str=None, da: str=None, checkproxydata: str=None, parse: str=None, ip: str=None, registrarrawtext: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Whois api v1 with password authentication."
    domainname: input domain name to lookup
        ipwhois: 1 results in returning the whois record for the hosting ip if the whois record for the tld of the input domain is not supported. Default: 0
        preferfresh: 1 results in getting the latest Whois record even if it's incomplete. Default: 0
        ignorerawtexts: 1 results in stripping all raw text from the output. Default: 0
        thinwhois: 0 | 1 (defaults to 0); 1 results in returning whois data from registry only, without fetching data from registrar; returned registry data corresponds to the WhoisRecord → registryData schema element
        callback: A javascript function used when the outputFormat is JSON; this is an implementation known as JSONP which invokes the callback on the returned response.
        registryrawtext: a string representing the registry whois raw text to be parsed; works only when the _parse parameter is equal to 1
        outputformat: XML | JSON (defaults to XML)
        da: 0 | 1 | 2 (defaults to 0) 1 results in a quick check on domain availability, 2 is slower but more accurate
        checkproxydata: 0 | 1 (defaults to 0); 1 results in fetching proxy/whois guard data, if it exists, in the WhoisRecord → privateWhoisProxy schema element
        parse: 0 | 1 (defaults to 0); 1 provides parsing for input whois raw texts described at the *RawText parameters
        ip: 0 | 1 (defaults to 0); 1 results in returning ips for the domain name
        registrarrawtext: a string representing the registrar whois raw text to be parsed; works only when the _parse parameter is equal to 1
        
    """
    url = f"https://whoisapi.p.rapidapi.com/whoisserver/WhoisService"
    querystring = {'domainname': domainname, }
    if ipwhois:
        querystring['ipWhois'] = ipwhois
    if preferfresh:
        querystring['preferFresh'] = preferfresh
    if ignorerawtexts:
        querystring['ignoreRawTexts'] = ignorerawtexts
    if thinwhois:
        querystring['thinWhois'] = thinwhois
    if callback:
        querystring['callback'] = callback
    if registryrawtext:
        querystring['registryRawText'] = registryrawtext
    if outputformat:
        querystring['outputformat'] = outputformat
    if da:
        querystring['da'] = da
    if checkproxydata:
        querystring['checkProxyData'] = checkproxydata
    if parse:
        querystring['_parse'] = parse
    if ip:
        querystring['ip'] = ip
    if registrarrawtext:
        querystring['registrarRawText'] = registrarrawtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

