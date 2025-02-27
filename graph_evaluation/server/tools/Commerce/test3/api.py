import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ddd(ddd: str=None, dddd: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dddd"
    ddd: dddd
        dddd: ddd
        
    """
    url = f"https://test3101.p.rapidapi.com/teteeee"
    querystring = {}
    if ddd:
        querystring['ddd'] = ddd
    if dddd:
        querystring['dddd'] = dddd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test3101.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

