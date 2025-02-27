import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_address_transactions(network: str='BSC_TESTNET', limit: str='10', offset: str='0', monitoraddressid: str='8485d9c3-7f52-4ba7-8ec2-41543effa6ae', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get address transactions"
    
    """
    url = f"https://address-monitor.p.rapidapi.com/address-transaction"
    querystring = {}
    if network:
        querystring['network'] = network
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if monitoraddressid:
        querystring['monitorAddressId'] = monitoraddressid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_service_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get service status"
    
    """
    url = f"https://address-monitor.p.rapidapi.com/service-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_address_transaction_webhook(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get address transaction webhook"
    
    """
    url = f"https://address-monitor.p.rapidapi.com/address-transaction/{transactionid}/webhook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quota_usage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quota usage"
    
    """
    url = f"https://address-monitor.p.rapidapi.com/rapidapi-quota-usage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_addresses(limit: str='100', offset: str='0', network: str='BSC_TESTNET', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get addresses"
    
    """
    url = f"https://address-monitor.p.rapidapi.com/address"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-monitor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

