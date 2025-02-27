import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(length: int, symbols: str=None, uppercase: str=None, numbers: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quickly generate a secure password."
    length: Max: 100
        symbols: Include symbols.
Default: True
        uppercase: Include uppercase letters.
Default: True
        numbers: Include numbers.
Default: True
        
    """
    url = f"https://secure-password-generator1.p.rapidapi.com/"
    querystring = {'length': length, }
    if symbols:
        querystring['symbols'] = symbols
    if uppercase:
        querystring['uppercase'] = uppercase
    if numbers:
        querystring['numbers'] = numbers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-password-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

