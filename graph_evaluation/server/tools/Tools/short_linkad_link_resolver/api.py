import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bypass(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main endpoint of the API. Here, you can submit a shortened URL to resolve it into the destination URL. If there is a chain of shortened URLs, it will recursively resolve them to reach the destination. Note that each resolve counts towards your quota."
    
    """
    url = f"https://short-link-ad-link-resolver.p.rapidapi.com/bypass"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "short-link-ad-link-resolver.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

