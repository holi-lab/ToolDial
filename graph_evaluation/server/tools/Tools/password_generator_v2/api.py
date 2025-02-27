import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def password_generator_api(length: int, character_types: str='uppercase', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Welcome to the password generator API! This API allows you to generate strong, random passwords for use in online accounts or other applications."
    length: The length of the password to generate (integer, required).
        character_types: The types of characters to include in the password (string, optional). Possible values are uppercase, lowercase, digits, and punctuation. If not specified, all character types will be included.
        
    """
    url = f"https://password-generator13.p.rapidapi.com/random_password"
    querystring = {'length': length, }
    if character_types:
        querystring['character_types'] = character_types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

