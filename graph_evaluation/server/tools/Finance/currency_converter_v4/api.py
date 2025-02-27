import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency(from_currency: str, amount: int, to_currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows users to convert a specified amount from one currency to another using real-time exchange rates. The API is accurate, easy to use, and can be integrated into any web or mobile application."
    
    """
    url = f"https://currency-converter229.p.rapidapi.com/currency_converter"
    querystring = {'from_currency': from_currency, 'amount': amount, 'to_currency': to_currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter229.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of All Currencies"
    
    """
    url = f"https://currency-converter229.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter229.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

