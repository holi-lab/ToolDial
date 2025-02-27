import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_nft_collections(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An Endpoint to get NFT Collections on Pancakeswap"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/nft/collections"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_price(token: str='0xe9e7cea3dedca5984780bafc599bd69add087d56', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Price"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/price"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_balance(address: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Balance"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/balance"
    querystring = {'address': address, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_decimal(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Decimal"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/decimals"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_supply(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Supply"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/supply"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_symbol(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Symbol"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/symbol"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_name(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Token Name"
    
    """
    url = f"https://pancakeswap-api-freemium.p.rapidapi.com/api/name"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pancakeswap-api-freemium.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

