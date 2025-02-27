import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def simbrief_get_latest_ofp(username: str, json: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest ofp in xml (default) or json based on username
		the json param must be 0 or 1"
    
    """
    url = f"https://simbrief-get-latest-ofp.p.rapidapi.com/"
    querystring = {'username': username, }
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simbrief-get-latest-ofp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

