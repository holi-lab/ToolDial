import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_flag_by_country_country_name(country_name: str='France', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flag emoji by its country code"
    
    """
    url = f"https://emojiflags.p.rapidapi.com/getflagbyname"
    querystring = {}
    if country_name:
        querystring['country_name'] = country_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emojiflags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_flag_by_country_code(code: str='FR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flag emoji by its country code"
    
    """
    url = f"https://emojiflags.p.rapidapi.com/getflagbycode"
    querystring = {}
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emojiflags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

