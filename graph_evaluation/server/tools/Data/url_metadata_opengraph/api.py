import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def parse(url: str, includesource: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: Url / Endpoint
        includesource: Include downloaded website html source
        
    """
    url = f"https://url-metadata-opengraph.p.rapidapi.com/parse"
    querystring = {'url': url, }
    if includesource:
        querystring['includeSource'] = includesource
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-metadata-opengraph.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

