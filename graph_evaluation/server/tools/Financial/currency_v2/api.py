import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange(base: str, to: str, int: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The currency converter service that converts the entered rate into the desired exchange rate."
    base: Currency Code (USD, EUR etc.)
        to: Target Currency Code
        int: Amount of Money (default: 1)
        
    """
    url = f"https://currency23.p.rapidapi.com/exchange"
    querystring = {'base': base, 'to': to, }
    if int:
        querystring['int'] = int
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cripto(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bitcoin and Altcoins prices, changes in the instant service."
    
    """
    url = f"https://currency23.p.rapidapi.com/cripto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencytoall(base: str, int: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that converts the entered exchange rate to other currencies."
    base: Currency Code (USD,EUR etc.)
        int: Amount of Money (default 1)
        
    """
    url = f"https://currency23.p.rapidapi.com/currencyToAll"
    querystring = {'base': base, }
    if int:
        querystring['int'] = int
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that brings the names and symbols of exchange rates."
    
    """
    url = f"https://currency23.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

