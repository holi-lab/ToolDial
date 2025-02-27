import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def usd(tocurrency: str, amount: str, fromcurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple Exchange USD"
    
    """
    url = f"https://currency-converter48.p.rapidapi.com/"
    querystring = {'toCurrency': tocurrency, 'amount': amount, 'fromCurrency': fromcurrency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter48.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gbp_cc(tocurrency: str, fromcurrency: str, amount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple British Pound"
    
    """
    url = f"https://currency-converter48.p.rapidapi.com/"
    querystring = {'toCurrency': tocurrency, 'fromCurrency': fromcurrency, 'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter48.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_currency(tocurrency: str, fromcurrency: str, amount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple converter"
    
    """
    url = f"https://currency-converter48.p.rapidapi.com/"
    querystring = {'toCurrency': tocurrency, 'fromCurrency': fromcurrency, 'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter48.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

