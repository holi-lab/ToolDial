import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def excahnge_currency(amount: str='100', fromcurrency: str='NGN', tocurrency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A simple API that allows you to exchanhge a specific amount from one currency to another."
    
    """
    url = f"https://currency-converter131.p.rapidapi.com/"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    if fromcurrency:
        querystring['fromCurrency'] = fromcurrency
    if tocurrency:
        querystring['toCurrency'] = tocurrency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter131.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

