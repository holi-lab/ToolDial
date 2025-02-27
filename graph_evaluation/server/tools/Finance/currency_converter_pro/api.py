import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_rates(base: str, currencies: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real-time exchange rate data of 170 world currencies. Updated every 10min."
    currencies: You can pass individual currencies for faster response.
Example:

EUR,RSD,AUD,CAD
        
    """
    url = f"https://currency-converter-pro1.p.rapidapi.com/latest-rates"
    querystring = {'base': base, }
    if currencies:
        querystring['currencies'] = currencies
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-pro1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported currencies."
    
    """
    url = f"https://currency-converter-pro1.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-pro1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_currency(amount: int, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert any amount from one currency to another."
    
    """
    url = f"https://currency-converter-pro1.p.rapidapi.com/convert"
    querystring = {'amount': amount, 'from': is_from, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-pro1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

