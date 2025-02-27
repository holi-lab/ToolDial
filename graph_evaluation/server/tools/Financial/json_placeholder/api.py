import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_currency(fromcurrency: str, amount: str, tocurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A simple financial API that allows you to change a specific amount from one currency to another"
    
    """
    url = f"https://json-placeholder155.p.rapidapi.com/"
    querystring = {'fromCurrency': fromcurrency, 'amount': amount, 'toCurrency': tocurrency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "json-placeholder155.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

