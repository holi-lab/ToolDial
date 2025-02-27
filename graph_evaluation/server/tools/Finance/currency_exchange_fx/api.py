import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def converter(from_currency: str, to_currency: str, from_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the converted amount in a specified currency."
    from_currency: The currency symbol from which the amount needs to be converted
        to_currency: The Currency symbol in which the amount needs to be converted
        from_value: The Amount to be converted in base currency
        
    """
    url = f"https://currency-exchange-fx.p.rapidapi.com/convert"
    querystring = {'from_currency': from_currency, 'to_currency': to_currency, 'from_value': from_value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange-fx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

