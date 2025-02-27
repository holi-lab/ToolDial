import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validator(ccvalidate: int=374648292645680, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "you can also check if given credit card number is valid real one"
    
    """
    url = f"https://random-fake-credit-card-details-and-validator.p.rapidapi.com/"
    querystring = {}
    if ccvalidate:
        querystring['ccvalidate'] = ccvalidate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-fake-credit-card-details-and-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generator(num: int=50, ccdetails: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "you can fetch credit card details + the number of the result (min:1 max:100)"
    
    """
    url = f"https://random-fake-credit-card-details-and-validator.p.rapidapi.com/"
    querystring = {}
    if num:
        querystring['num'] = num
    if ccdetails:
        querystring['ccdetails'] = ccdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-fake-credit-card-details-and-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

