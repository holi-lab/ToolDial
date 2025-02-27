import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_device_asset_by_identifier_copy(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "External info Device-Asset by Identifier"
    
    """
    url = f"https://basic-info-vikkon-assets.p.rapidapi.com/apivikkon-sandbox.herokuapp.com/api/v1/external-system/{identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basic-info-vikkon-assets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_device_asset_by_identifier(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "External info Device-Asset by Identifier"
    
    """
    url = f"https://basic-info-vikkon-assets.p.rapidapi.com/apivikkon-sandbox.herokuapp.com/api/v1/external-system/{identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basic-info-vikkon-assets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

