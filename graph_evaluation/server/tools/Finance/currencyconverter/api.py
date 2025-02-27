import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a list of supported currencies"
    
    """
    url = f"https://currencyconverter9.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_one(to: str, is_from: str='eur', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a single currency exchange rate"
    to: Target currency 3-letter symbol
        is_from: Base currency 3-letter symbol, defaults to USD
        
    """
    url = f"https://currencyconverter9.p.rapidapi.com/fetch-one"
    querystring = {'to': to, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all(is_from: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all available currency rates"
    is_from: Base currency 3-letter symbol, defaults to USD
        
    """
    url = f"https://currencyconverter9.p.rapidapi.com/fetch-all"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_multi(to: str, is_from: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch multiple currency rates at once"
    to: Target currencies, comma separated list of 3-letter symbols
        is_from: Base currency 3-letter symbol, defaults to USD
        
    """
    url = f"https://currencyconverter9.p.rapidapi.com/fetch-multi"
    querystring = {'to': to, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical(date: str, is_from: str='USD', to: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch exchange rates for a date in the past"
    date: UTC date in YYYY-MM-DD format. Must be in the past and within the last 150 days.
        is_from: Base currency 3-letter symbol, defaults to USD
        to: Target currencies, comma separated list of 3-letter symbols. Defaults to all currencies if omitted
        
    """
    url = f"https://currencyconverter9.p.rapidapi.com/historical"
    querystring = {'date': date, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(to: str, amount: int, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an amount of one currency into another currency"
    to: Target currency 3-letter symbol
        amount: Amount of source currency to convert
        is_from: Base currency 3-letter symbol, defaults to USD
        
    """
    url = f"https://currencyconverter9.p.rapidapi.com/convert"
    querystring = {'to': to, 'amount': amount, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyconverter9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

