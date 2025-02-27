import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Available currencies codes"
    
    """
    url = f"https://currency-converter139.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter139.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_currency_data(base: str='USD', date: str='2020-01-01', symbols: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the rates on the specific date"
    base: Base currency code (AUD, CAD, EUR, GBP …)
        date: DATE (YYYY-MM-DD)
        symbols: Currencies filter
        
    """
    url = f"https://currency-converter139.p.rapidapi.com/historical"
    querystring = {}
    if base:
        querystring['base'] = base
    if date:
        querystring['date'] = date
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter139.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_currencies(base: str='USD', symbols: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the latest exchange rates for the base currency code you have supplied."
    base: Base currency code (AUD, CAD, EUR, GBP …)
        symbols: Currencies filter
        
    """
    url = f"https://currency-converter139.p.rapidapi.com/latest"
    querystring = {}
    if base:
        querystring['base'] = base
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter139.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_converter(date: str='2020-01-01', base: str='USD', symbols: str=None, value: int=99, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts from one currency to another"
    date: DATE (YYYY-MM-DD)
        base: Base currency code (AUD, CAD, EUR, GBP …)
        symbols: Currencies filter
        value: Value that needs to be converted
        
    """
    url = f"https://currency-converter139.p.rapidapi.com/convert"
    querystring = {}
    if date:
        querystring['date'] = date
    if base:
        querystring['base'] = base
    if symbols:
        querystring['symbols'] = symbols
    if value:
        querystring['value'] = value
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter139.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

