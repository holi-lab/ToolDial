import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_currency(amount: str, fromcurrency: str, tocurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API that generates the required Exchange Rate through parameters specified. It shows recommended countries that use that same exchange rate.
		Eg: specify the currency code you want to convert from and the currency code you want to convert to and also the amount.
		USD SLL 20"
    
    """
    url = f"https://currency-conveter1.p.rapidapi.com/"
    querystring = {'amount': amount, 'fromCurrency': fromcurrency, 'toCurrency': tocurrency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

