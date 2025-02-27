import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def proxy(url: str, headers: str=None, query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Proxy your request anywhere."
    url: The url you want to visit
        headers: The headers you want to set. Make sure they are in json format.
        query: The query you want to set. Make sure they are in json format.
        
    """
    url = f"https://proxy12.p.rapidapi.com/proxy"
    querystring = {'url': url, }
    if headers:
        querystring['headers'] = headers
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxy12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

