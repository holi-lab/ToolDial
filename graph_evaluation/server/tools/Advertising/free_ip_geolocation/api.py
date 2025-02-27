import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def multi_language_support(ip: str, language: str, api_key: str='test', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an IP's country name in a different language. 5 languages are currently supported i.e. English - en, German - de, French - fr, Japanese - ja and Simplified Chinese -zh-CN. The response will contain an ASCII encoded version of the country name in the language you specify."
    api_key: Use 'test' for highly rate-limited functions. Get your own API-key here: https://ipdata.co/
        
    """
    url = f"https://jkosgei-free-ip-geolocation-v1.p.rapidapi.com/{ip}/{language}"
    querystring = {}
    if api_key:
        querystring['api-key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jkosgei-free-ip-geolocation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def any_ip(ip: str, api_key: str='test', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the location, city, country, lat/long etc of any IP address"
    api_key: Use 'test' for highly rate-limited functions. Get your own API-key here: https://ipdata.co/
        
    """
    url = f"https://jkosgei-free-ip-geolocation-v1.p.rapidapi.com/{ip}"
    querystring = {}
    if api_key:
        querystring['api-key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jkosgei-free-ip-geolocation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

