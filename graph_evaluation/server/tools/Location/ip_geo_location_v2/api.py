import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def visitor_lookup(language: str=None, filter: str=None, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the IP address of the client with all the location data"
    language: language code to return the results in the specific language. Available options are: en, ru, zh, es, ar, fr, fa, ja, pl, it, pt, de
        filter: Lets you to return only required data. It can be comma separated. Options: asn, city, country, continent, area, currency, security, time, postcode. If left blank the API will return all available data.
        format: The desired format of the data.  Options: json or xml
        
    """
    url = f"https://ip-geo-location.p.rapidapi.com/ip/check"
    querystring = {}
    if language:
        querystring['language'] = language
    if filter:
        querystring['filter'] = filter
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geo-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_lookup(ip: str, format: str='json', language: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides geo information for the given IP"
    ip: IPv4 or IPv6 address
        format: The desired format of the data.  Options: json or xml
        language: language code to return the results in the specific language. Available options are: en, ru, zh, es, ar, fr, fa, ja, pl, it, pt, de
        filter: Lets you to return only required data. It can be comma separated. Options: asn, city, country, continent, area, currency, security, time, postcode. If left blank the API will return all available data.
        
    """
    url = f"https://ip-geo-location.p.rapidapi.com/ip/{ip}"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geo-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

