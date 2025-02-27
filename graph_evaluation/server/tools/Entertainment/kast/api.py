import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def info_news(q: str, x_rapidapi_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "official api"
    
    """
    url = f"https://kast1.p.rapidapi.com/info/news"
    querystring = {'q': q, }
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kast1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

