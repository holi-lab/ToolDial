import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_password(length: int=8, uppercases: bool=None, lowercases: bool=None, digits: bool=None, specials: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get endpoint for generating a new password."
    
    """
    url = f"https://password-generator8.p.rapidapi.com/newpass/"
    querystring = {}
    if length:
        querystring['length'] = length
    if uppercases:
        querystring['uppercases'] = uppercases
    if lowercases:
        querystring['lowercases'] = lowercases
    if digits:
        querystring['digits'] = digits
    if specials:
        querystring['specials'] = specials
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

