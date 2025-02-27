import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest(flex: bool=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "last P2000 message
		
		- `?limit=10` (number between 10 and 500)
		- `?flex=true` (raw FLEX line with date)"
    flex: Enable if you only want the raw FLEX line
        limit: Limit the amount of p2000 messages you want to show
        
    """
    url = f"https://p2000.p.rapidapi.com/latest"
    querystring = {}
    if flex:
        querystring['flex'] = flex
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "p2000.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

