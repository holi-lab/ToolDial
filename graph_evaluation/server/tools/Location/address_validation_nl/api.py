import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_addresses(postcode: str, housenumber: int, housenumbersuffix: str='B', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Address"
    
    """
    url = f"https://address-validation-nl1.p.rapidapi.com/api/v1/addresses"
    querystring = {'Postcode': postcode, 'HouseNumber': housenumber, }
    if housenumbersuffix:
        querystring['HouseNumberSuffix'] = housenumbersuffix
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-nl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_addresses(postcode: str, housenumber: int, housenumbersuffix: str='B', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Address"
    
    """
    url = f"https://address-validation-nl1.p.rapidapi.com/api/v1/addresses"
    querystring = {'Postcode': postcode, 'HouseNumber': housenumber, }
    if housenumbersuffix:
        querystring['HouseNumberSuffix'] = housenumbersuffix
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation-nl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

