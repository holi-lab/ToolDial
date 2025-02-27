import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_currency_data(currencies: str='EUR,GBP', base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest rates"
    currencies: Limit results to specific currencies (comma-separated list of 3-letter codes) [optional]
        base: The base currency (3-letter code, default: USD) [optional]
        
    """
    url = f"https://currency-converter28.p.rapidapi.com/latest"
    querystring = {}
    if currencies:
        querystring['currencies'] = currencies
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_converter(amount: int, to: str, is_from: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts from one currency to another"
    amount: The value to be converted
        to: The target currency (3-letter code)
        is_from: The base currency (3-letter code, default USD) [optional]
        
    """
    url = f"https://currency-converter28.p.rapidapi.com/convert"
    querystring = {'amount': amount, 'to': to, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of available currencies"
    
    """
    url = f"https://currency-converter28.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

