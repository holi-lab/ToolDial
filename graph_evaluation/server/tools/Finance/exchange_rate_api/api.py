import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_rates(base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest foreign exchange reference rates.Latest endpoint will return exchange rate data updates every 1 minutes."
    base: the code you want to use as a base currency.
        
    """
    url = f"https://exchange-rate-api1.p.rapidapi.com/latest"
    querystring = {}
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists the currencies that we support."
    
    """
    url = f"https://exchange-rate-api1.p.rapidapi.com/codes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_currency(target: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the endpoint is useful for applications where you just want to convert between two specific currencies."
    target: The three-letter currency code of the currency you would like to convert to.
        base: The three-letter currency code of the currency you would like to convert from.
        
    """
    url = f"https://exchange-rate-api1.p.rapidapi.com/convert"
    querystring = {'target': target, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchange-rate-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

