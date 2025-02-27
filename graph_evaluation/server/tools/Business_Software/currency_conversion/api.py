import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_currency(to: str=None, is_from: str=None, amount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert Currency"
    
    """
    url = f"https://currency-conversion3.p.rapidapi.com/latest"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

