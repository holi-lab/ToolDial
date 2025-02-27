import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currency_converter(language: str=None, is_from: str='AUD', amount: int=1, format: str='json', to: str='CAD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts from one currency to another"
    is_from: Base currency code (AUD, CAD, EUR, GBP ...)
        amount: Amount that needs to be converted
        format: Response format: json or xml
        to: Desired currency code (AUD, CAD, EUR, GBP ...)
        
    """
    url = f"https://currency-converter5.p.rapidapi.com/currency/convert"
    querystring = {}
    if language:
        querystring['language'] = language
    if is_from:
        querystring['from'] = is_from
    if amount:
        querystring['amount'] = amount
    if format:
        querystring['format'] = format
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_currency_data(date: str, amount: int=1, to: str='GBP', is_from: str='EUR', format: str='json', language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns the rates on the specific date"
    
    """
    url = f"https://currency-converter5.p.rapidapi.com/currency/historical/{date}"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_currencies(format: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of available currencies"
    format: Response format: json or xml
        
    """
    url = f"https://currency-converter5.p.rapidapi.com/currency/list"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

