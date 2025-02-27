import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert(is_from: str, to: str, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When this endpoint is called, it uses the 'from' and 'to' query parameters to determine the currencies to convert from and to. It also uses the 'value' query parameter to determine the amount of currency to convert."
    
    """
    url = f"https://currency-conversion-api1.p.rapidapi.com/convert"
    querystring = {'from': is_from, 'to': to, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

