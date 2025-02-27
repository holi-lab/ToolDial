import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def terminal_delivery(authorization: str='sk_live_bm4KIzVbJEgYpmIULgLZZJYyXR0EXRnc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DELIVER GOODS"
    
    """
    url = f"https://terminal7.p.rapidapi.com/api.terminal.africa/v1/webhooks"
    querystring = {}
    if authorization:
        querystring['AUTHORIZATION'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "terminal7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

