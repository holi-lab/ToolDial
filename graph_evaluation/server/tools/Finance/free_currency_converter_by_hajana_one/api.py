import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currency_api_php(amount: int, is_from: str, to: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert Ant Currency"
    amount: Amount you want to convert
        from: Three Letters of currency
        to: Three Letters of Currency
        
    """
    url = f"https://hajana1-free-currency-converter-by-hajana-one-v1.p.rapidapi.com/currency-api.php"
    querystring = {'amount': amount, 'from': is_from, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hajana1-free-currency-converter-by-hajana-one-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

