import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def predict(period: int, crypto: str, history: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the future prices of cryptocurrency"
    
    """
    url = f"https://cryptorch.p.rapidapi.com/api/v2"
    querystring = {'period': period, 'crypto': crypto, }
    if history:
        querystring['history'] = history
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptorch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

