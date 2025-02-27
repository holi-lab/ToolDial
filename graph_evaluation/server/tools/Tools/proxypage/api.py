import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_proxy(content_type: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random proxy,
		
		choose type and country"
    type: HTTP, HTTPS, SOCKS4, SOCKS5, CONNECT:25
        country: You can specify a country for a proxy that you want to be returened

        
    """
    url = f"https://proxypage1.p.rapidapi.com/v1/tier1random"
    querystring = {'Content-Type': content_type, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxypage1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tier2(type: str, ssl: str=None, limit: int=100, is_anonymous: str=None, country: str='US', latency: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tier 2 proxies
		
		Each proxy returned costs 1 credit
		
		
		With our /v1/tier2 endpoint you can set different parameters for proxies that you need.
		
		You can set type which is just your proxy type, either HTTP, HTTPS, SOCKS4, SOCKS5, CONNECT:25 (which is smtp proxy)
		
		for limit set an integer that will tell us how many proxies you will need. Our users usually set a limit to avoid using too many credits.
		
		With latency you can set an integer which will filter out all proxies that have a latency higher then specified.
		
		ssl is a boolean parameter, you can filter out proxies that support ssl or don't
		
		is_anonymous is also a boolean statemet where you can filter anonymous proxies
		
		country is a parameter that you can use to set a country that you want."
    
    """
    url = f"https://proxypage1.p.rapidapi.com/v1/tier2"
    querystring = {'type': type, }
    if ssl:
        querystring['ssl'] = ssl
    if limit:
        querystring['limit'] = limit
    if is_anonymous:
        querystring['is_anonymous'] = is_anonymous
    if country:
        querystring['country'] = country
    if latency:
        querystring['latency'] = latency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxypage1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tier1(content_type: str, latency: int=None, country: str='US', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List our tier 1 proxies with filters
		This proxies are more comprehensively checked
		
		
		You can set type which is just your proxy type, either HTTP, HTTPS
		
		for limit set an integer that will tell us how many proxies you will need. Our users usually set a limit to avoid using too many credits.
		
		With latency you can set an integer which will filter out all proxies that have a latency higher then specified.
		
		ssl is a boolean parameter, you can filter out proxies that support ssl or don't
		
		is_anonymous is also a boolean statemet where you can filter anonymous proxies
		
		country is a parameter that you can use to set a country that you want."
    type: HTTP, HTTPS, SOCKS4, SOCKS5, CONNECT:25
        latency: ms latency for a proxy, everything that is below that value is returned

        country: You can specify a country for a proxy that you want to be returened

        limit: Limit the number of proxies returned, helps you control how many credits can be used

        
    """
    url = f"https://proxypage1.p.rapidapi.com/v1/tier1"
    querystring = {'Content-Type': content_type, }
    if latency:
        querystring['latency'] = latency
    if country:
        querystring['country'] = country
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxypage1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

