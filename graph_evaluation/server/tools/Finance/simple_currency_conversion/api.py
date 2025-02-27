import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency(to: str, is_from: str='USD', amount: int=1, format: str='json', decrease: int=0, increase: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert from one currency to another"
    to: Target ISO 4217 currency code, ex: EUR
        is_from: Source ISO 4217 currency code. Default Value: USD
        amount: Amount of the source currency to convert. Default Value: 1
        format: Output format, text or json. Default Value: text
        decrease: Decrease the unit exchange rate by the specified ammount. Useful to make minor adjustments to the exchange rate. Default Value: 0
        increase: Increase the unit exchange rate by the specified ammount. Useful to make minor adjustments to the exchange rate. Default Value: 0
        
    """
    url = f"https://simple-currency-conversion.p.rapidapi.com/api/method/exchangerate"
    querystring = {'to': to, }
    if is_from:
        querystring['from'] = is_from
    if amount:
        querystring['amount'] = amount
    if format:
        querystring['format'] = format
    if decrease:
        querystring['decrease'] = decrease
    if increase:
        querystring['increase'] = increase
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-currency-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

