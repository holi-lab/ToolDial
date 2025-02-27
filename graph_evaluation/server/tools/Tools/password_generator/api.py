import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_password(excludesimilarcharacters: bool=None, strict: bool=None, uppercase: bool=None, exclude: str='1', symbols: bool=None, lowercase: bool=None, length: int=15, numbers: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Password"
    
    """
    url = f"https://password-generator18.p.rapidapi.com/generate-password"
    querystring = {}
    if excludesimilarcharacters:
        querystring['excludeSimilarCharacters'] = excludesimilarcharacters
    if strict:
        querystring['strict'] = strict
    if uppercase:
        querystring['uppercase'] = uppercase
    if exclude:
        querystring['exclude'] = exclude
    if symbols:
        querystring['symbols'] = symbols
    if lowercase:
        querystring['lowercase'] = lowercase
    if length:
        querystring['length'] = length
    if numbers:
        querystring['numbers'] = numbers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

