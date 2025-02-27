import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def proxy(lasttested: int=None, ip: str=None, anonymity: str=None, protocol: str=None, allowsrefererheader: bool=None, allowsuseragentheader: bool=None, allowscustomheaders: bool=None, allowspost: bool=None, allowshttps: bool=None, country: str=None, notcountry: str=None, port: str=None, allowscookies: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For more indepth info on the Proxy API see docs here https://getproxylist.com"
    lasttested: Seconds since last tested
        ip: The IP of the proxy
        anonymity: How much of your info is hidden
        protocol: The type of proxy
        allowsrefererheader: Supports the Referer header
        allowsuseragentheader: Supports the User-Agent headers
        allowscustomheaders: Supports any custom headers
        allowspost: Supports POST requests
        allowshttps: Supports HTTPS requests
        country: Country the proxy originates from
        notcountry: Filter out specific countries
        port: port
        allowscookies: Supports cookies
        
    """
    url = f"https://getproxylist-getproxylist-v1.p.rapidapi.com/proxy"
    querystring = {}
    if lasttested:
        querystring['lastTested'] = lasttested
    if ip:
        querystring['ip'] = ip
    if anonymity:
        querystring['anonymity'] = anonymity
    if protocol:
        querystring['protocol'] = protocol
    if allowsrefererheader:
        querystring['allowsRefererHeader'] = allowsrefererheader
    if allowsuseragentheader:
        querystring['allowsUserAgentHeader'] = allowsuseragentheader
    if allowscustomheaders:
        querystring['allowsCustomHeaders'] = allowscustomheaders
    if allowspost:
        querystring['allowsPost'] = allowspost
    if allowshttps:
        querystring['allowsHttps'] = allowshttps
    if country:
        querystring['country'] = country
    if notcountry:
        querystring['notCountry'] = notcountry
    if port:
        querystring['port'] = port
    if allowscookies:
        querystring['allowscookies'] = allowscookies
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getproxylist-getproxylist-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

