import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_shortner(url: str, signature: str='c808b5e03a', action: str='shorturl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "URL Shortner"
    
    """
    url = f"https://0-1-url-shortner.p.rapidapi.com/yourls-api.php"
    querystring = {'url': url, }
    if signature:
        querystring['signature'] = signature
    if action:
        querystring['action'] = action
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "0-1-url-shortner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

