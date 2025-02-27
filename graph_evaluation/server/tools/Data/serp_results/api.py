import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def serp(query: str, proxy: str='US', max: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SERP"
    query: Search Query

e.g. `music site:youtube.com`
        proxy: Proxy Country
        max: Number of Results
        
    """
    url = f"https://serp-results.p.rapidapi.com/"
    querystring = {'query': query, }
    if proxy:
        querystring['proxy'] = proxy
    if max:
        querystring['max'] = max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serp-results.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

