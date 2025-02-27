import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_currency(amount: str, tocurrency: str, fromcurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A simple API that allows you to exchange a specific amount from one currency to another."
    
    """
    url = f"https://currency-converter80.p.rapidapi.com/"
    querystring = {'amount': amount, 'toCurrency': tocurrency, 'fromCurrency': fromcurrency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter80.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

