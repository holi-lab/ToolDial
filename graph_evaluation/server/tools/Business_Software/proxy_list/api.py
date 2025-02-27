import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def proxy_get(type: str='http', country: str='US', anonymity: str='high', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of proxies based on the given parameters. The list is updated every 1 minute. API returns only working at the time of check proxies."
    type: Possible values: http, https, socks4, socks5
        country: Use country code to filter data by country. Please check https://www.nationsonline.org/oneworld/country_code_list.htm for more information about country codes.
        anonymity: Possible values: high, anonymous, transparent
        
    """
    url = f"https://proxy-list2.p.rapidapi.com/proxy/get"
    querystring = {}
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    if anonymity:
        querystring['anonymity'] = anonymity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxy-list2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def proxy_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ReturnS the health status of the API. Also return last time the proxies check was completed. Time is in UTC."
    
    """
    url = f"https://proxy-list2.p.rapidapi.com/proxy/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxy-list2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

