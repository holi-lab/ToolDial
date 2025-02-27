import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_currency(tocurrency: str, fromcurrency: str, amount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "這是一個可以將輸入的金額幣別轉成對應幣別的API"
    
    """
    url = f"https://currency-converter-test.p.rapidapi.com/"
    querystring = {'toCurrency': tocurrency, 'fromCurrency': fromcurrency, 'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-test.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

