import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shorten(longurl: str='http://www.mashape.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shortens a url"
    longurl: a long url
        
    """
    url = f"https://akshatarora-bitly-example-v1.p.rapidapi.com/v3/shorten/"
    querystring = {}
    if longurl:
        querystring['longUrl'] = longurl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "akshatarora-bitly-example-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

