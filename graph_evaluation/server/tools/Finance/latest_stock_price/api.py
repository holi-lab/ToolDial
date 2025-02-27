import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price(indices: str, identifier: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch latest stock price based on indices"
    indices: Define Stock Indices
        identifier: Define Stock Identifier
Supports multiple comma separated Identifier
        
    """
    url = f"https://latest-stock-price.p.rapidapi.com/price"
    querystring = {'Indices': indices, }
    if identifier:
        querystring['Identifier'] = identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices(indices: str, identifier: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch latest stock price based on indices"
    indices: Define Stock Indices
        identifier: Define Stock Identifier
Supports multiple comma separated Identifier
        
    """
    url = f"https://latest-stock-price.p.rapidapi.com/price"
    querystring = {'Indices': indices, }
    if identifier:
        querystring['Identifier'] = identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_all(identifier: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch latest stock price"
    
    """
    url = f"https://latest-stock-price.p.rapidapi.com/any"
    querystring = {}
    if identifier:
        querystring['Identifier'] = identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-stock-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

