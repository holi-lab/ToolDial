import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scan(factory_address: str, exchange: str, router_address: str, token_b: str, chain: str, token_a: str, check_contract_creation: str='1', check_liquidity_lock: str='1', check_holders: str='1', check_smart_contract: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Honeypot Detection"
    
    """
    url = f"https://honeypotapi.p.rapidapi.com/api/v1/scan/"
    querystring = {'factory_address': factory_address, 'exchange': exchange, 'router_address': router_address, 'token_b': token_b, 'chain': chain, 'token_a': token_a, }
    if check_contract_creation:
        querystring['check_contract_creation'] = check_contract_creation
    if check_liquidity_lock:
        querystring['check_liquidity_lock'] = check_liquidity_lock
    if check_holders:
        querystring['check_holders'] = check_holders
    if check_smart_contract:
        querystring['check_smart_contract'] = check_smart_contract
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "honeypotapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges(chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Supported Exchanges"
    
    """
    url = f"https://honeypotapi.p.rapidapi.com/api/v1/exchanges/"
    querystring = {'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "honeypotapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Supported chain list"
    
    """
    url = f"https://honeypotapi.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "honeypotapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pairs(chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pair List"
    
    """
    url = f"https://honeypotapi.p.rapidapi.com/api/v1/pairs/"
    querystring = {'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "honeypotapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

