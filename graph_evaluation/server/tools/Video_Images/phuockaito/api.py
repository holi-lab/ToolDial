import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phuockaito(hinhanh: str='text', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "phuockaito"
    
    """
    url = f"https://phuockaito.p.rapidapi.com/phuockaitokid.freevnn.com/phuockaito"
    querystring = {}
    if hinhanh:
        querystring['HinhAnh'] = hinhanh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phuockaito.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phuockaito_copy(hinhanh: str='text', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "phuockaito"
    
    """
    url = f"https://phuockaito.p.rapidapi.com/phuockaitokid.freevnn.com/phuockaito"
    querystring = {}
    if hinhanh:
        querystring['HinhAnh'] = hinhanh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phuockaito.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

