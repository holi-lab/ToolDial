import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def history(callback: str, limit: str, subscribe_key: str, channel: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get older messages."
    callback: JSONP Callback
        limit: Limit of messages to return
        subscribe_key: Subscribe Key
        channel: Channel Name
        
    """
    url = f"https://pubnub-pubnub.p.rapidapi.com/history/{subscribe_key}/{channel}/{callback}/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pubnub-pubnub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time(callback: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Timetoken from PubNub Network"
    callback: JSONP Callback
        
    """
    url = f"https://pubnub-pubnub.p.rapidapi.com/time/{callback}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pubnub-pubnub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

