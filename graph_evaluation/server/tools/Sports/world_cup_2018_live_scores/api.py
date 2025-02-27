import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events_subscribed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the events subscribed. Not available for BASIC plan."
    
    """
    url = f"https://heisenbug-world-cup-2018-live-scores-v1.p.rapidapi.com/api/worldcup/push/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-world-cup-2018-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activate_webhook(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Activate a webhook registered with the subscribe endpoint. Not available for BASIC plan."
    token: Token
        
    """
    url = f"https://heisenbug-world-cup-2018-live-scores-v1.p.rapidapi.com/api/worldcup/push/activate"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heisenbug-world-cup-2018-live-scores-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

