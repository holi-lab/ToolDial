import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def paw_some_currency_trading(qry: str='100 EUR to USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Donate a bone, make a doggo's day."
    qry: Example 100 USD to EUR
or 100 EUR to USD
or 10 AUS to EUR
        
    """
    url = f"https://paw-some-currency-trading.p.rapidapi.com/convert"
    querystring = {}
    if qry:
        querystring['qry'] = qry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "paw-some-currency-trading.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

