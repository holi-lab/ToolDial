import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def password_generator(length: str, specials: str, lowercases: str, digits: str, uppercases: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Strong Password based on Parameters"
    
    """
    url = f"https://password-generator10.p.rapidapi.com/"
    querystring = {'length': length, 'specials': specials, 'lowercases': lowercases, 'digits': digits, 'uppercases': uppercases, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

