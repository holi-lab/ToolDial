import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random(length: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Random String with custom length
		BASIC: 20 
		PRO: 30
		ULTRA: 70
		MEGA: 150"
    
    """
    url = f"https://helper-function.p.rapidapi.com/string/random"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helper-function.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_uuid(timestamp_first: bool=None, remove_dash: bool=None, count: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate UUID v4 with dash or not.
		max
		BASIC: 3
		PRO: 20
		ULTRA: 50
		MEGA: 150"
    
    """
    url = f"https://helper-function.p.rapidapi.com/string/uuid"
    querystring = {}
    if timestamp_first:
        querystring['timestamp_first'] = timestamp_first
    if remove_dash:
        querystring['remove_dash'] = remove_dash
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helper-function.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

