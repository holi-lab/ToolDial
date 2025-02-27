import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def echo(header1: str='1', query2: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a GET request to the Echo"
    
    """
    url = f"https://simpleecho.p.rapidapi.com/api/echo"
    querystring = {}
    if header1:
        querystring['header1'] = header1
    if query2:
        querystring['query2'] = query2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpleecho.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

