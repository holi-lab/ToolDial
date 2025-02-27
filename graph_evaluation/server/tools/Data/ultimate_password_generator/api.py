import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_password(numbers: bool=None, pwcount: int=3, symbols: bool=None, length: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get your desired password with specific options to use"
    
    """
    url = f"https://ultimate-password-generator.p.rapidapi.com/"
    querystring = {}
    if numbers:
        querystring['numbers'] = numbers
    if pwcount:
        querystring['pwCount'] = pwcount
    if symbols:
        querystring['symbols'] = symbols
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ultimate-password-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

