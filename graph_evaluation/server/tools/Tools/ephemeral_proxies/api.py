import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_residential_proxies_service_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the current status of the service, including the total number of residential proxies available and grouped by country.
		"
    
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/residential/service_status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def obtain_a_new_residential_proxy(whitelist_ip: str=None, countries: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The connection proxy details are `proxy.host` and `proxy.port` from the response fields.
		The proxy will allow connections from the same source IP as the one making this API call. Make use of the `whitelist_ip` parameter to allow an additional ip.
		Countries can be selected by using the `countries` parameter, otherwise the country is randomly selected. A list of countries is available on *Service Status* endpoint.
		The proxy is available for 30 mins.
		"
    whitelist_ip: The proxy that this endpoint returns will allow connections from the source ip that is making this API call.
Additionally, you can allow an extra ip to connect to the proxy by setting this parameter to the ip that you would like to whitelist.
The list of ips that the proxy has whitelisted is returned in the API response.

        countries: List of countries, separate it by comma, that you would like the proxies to be located.
Example: GB,US,PL,RU

        
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/residential/proxy"
    querystring = {}
    if whitelist_ip:
        querystring['whitelist_ip'] = whitelist_ip
    if countries:
        querystring['countries'] = countries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_datacenter_proxies_service_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the current status of the service, including the total number of datacenter proxies available and grouped by country.
		"
    
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/datacenter/service_status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def obtain_a_new_datacenter_proxy(whitelist_ip: str=None, countries: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The connection proxy details are `proxy.host` and `proxy.port` from the response fields.
		The proxy will allow connections from the same source IP as the one making this API call. Make use of the `whitelist_ip` parameter to allow an additional ip.
		Countries can be selected by using the `countries` parameter, otherwise the country is randomly selected. A list of countries is available on *Service Status* endpoint.
		The proxy is available for 30 mins.
		"
    whitelist_ip: The proxy that this endpoint returns will allow connections from the source ip that is making this API call.
Additionally, you can allow an extra ip to connect to the proxy by setting this parameter to the ip that you would like to whitelist.
The list of ips that the proxy has whitelisted is returned in the API response.

        countries: List of countries, separate it by comma, that you would like the proxies to be located.
Example: GB,US,PL,RU

        
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/datacenter/proxy"
    querystring = {}
    if whitelist_ip:
        querystring['whitelist_ip'] = whitelist_ip
    if countries:
        querystring['countries'] = countries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_traffic_balance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the user's traffic balance for current monthly subscription.
		"
    
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/residential/balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extend_expiration_time_of_a_datacenter_proxy(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By calling this endpoint the expiration time of an already allocated proxy will be extended by 30 mins.
		Successive calls will keep incrementing the expiration time, up to a maximum of 24 hours.
		"
    id: The proxy identifier to extend the expiration time. This identifier can be obtained from the response of /v2/datacenter/proxy, field proxy.id.

        
    """
    url = f"https://ephemeral-proxies.p.rapidapi.com/v2/datacenter/extend_proxy"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ephemeral-proxies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

