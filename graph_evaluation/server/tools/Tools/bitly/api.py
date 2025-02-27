import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shorten(longurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a long URL, returns a bitly short URL."
    longurl: a long URL to be shortened (example: http://betaworks.com/).
        
    """
    url = f"https://ismaelc-bitly.p.rapidapi.com/v3/shorten"
    querystring = {'longUrl': longurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ismaelc-bitly.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

