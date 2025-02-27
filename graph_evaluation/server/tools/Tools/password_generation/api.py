import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_a_password_using_the_entropy_method(method: str, level: str='low', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints supports password generation using three levels of entropy: low, medium and high."
    
    """
    url = f"https://password-generation.p.rapidapi.com/api/v1/password"
    querystring = {'method': method, }
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_a_password_using_the_symbols_method(method: str, length: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint supports password generation with a  maximum length of 20 characters"
    
    """
    url = f"https://password-generation.p.rapidapi.com/api/v1/password"
    querystring = {'method': method, }
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

