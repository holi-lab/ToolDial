import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mars_test_chat(localhost6969: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a chat API. Users will plug in a server and start a chat room. no screen name required."
    localhost6969: this is the mars chain
        
    """
    url = f"https://chat-rooms.p.rapidapi.com/localhost6969"
    querystring = {}
    if localhost6969:
        querystring['localhost6969'] = localhost6969
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chat-rooms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

