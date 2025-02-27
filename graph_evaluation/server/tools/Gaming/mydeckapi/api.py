import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpilebysessionidandname(name: str, sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Give the session id and unique name of the pile you want to get information about"
    
    """
    url = f"https://mydeckapi2.p.rapidapi.com/api/v1/pile"
    querystring = {'name': name, 'sessionId': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mydeckapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayerbysessionidandname(name: str, sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Give the session id and unique name of the player you want to get information about"
    
    """
    url = f"https://mydeckapi2.p.rapidapi.com/api/v1/player"
    querystring = {'name': name, 'sessionId': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mydeckapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

