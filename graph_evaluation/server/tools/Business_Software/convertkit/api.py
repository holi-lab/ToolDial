import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def account(api_secret: str, api_key: str='SdqBPcY_TiBxmpf8I0XyUg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the current account"
    api_secret: Your ConvertKit account API secret
        api_key: Your ConvertKit account API key
        
    """
    url = f"https://convertkit.p.rapidapi.com/account"
    querystring = {'api_secret': api_secret, }
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convertkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

