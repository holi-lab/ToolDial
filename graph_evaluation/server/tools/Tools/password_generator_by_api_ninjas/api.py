import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_passwordgenerator(exclude_special_chars: bool=None, exclude_numbers: bool=None, length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Password Generator API endpoint. Returns a random password string adhering to the specified parameters."
    exclude_special_chars: Whether to exclude special characters(**!@#$%^&*()**) from the password. Must be either **true** or **false**. If not set, a default value of **false** will be used.
        exclude_numbers: Whether to exclude numbers from the password. Must be either **true** or **false**. If not set, a default value of **false** will be used.
        length: length of password in characters. If not set, a default value of **16** is used.
        
    """
    url = f"https://password-generator-by-api-ninjas.p.rapidapi.com/v1/passwordgenerator"
    querystring = {}
    if exclude_special_chars:
        querystring['exclude_special_chars'] = exclude_special_chars
    if exclude_numbers:
        querystring['exclude_numbers'] = exclude_numbers
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

