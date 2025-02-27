import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency(to: str, amount: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts an amount of one currency into that of another"
    
    """
    url = f"https://currency-rate-converter.p.rapidapi.com/convertcurrency"
    querystring = {'to': to, 'amount': amount, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-rate-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

