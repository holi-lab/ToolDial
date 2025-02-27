import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, gl: str=None, cr: str=None, tbs: str=None, num: int=20, start: int=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google search and get results"
    hl: Locale of Google search results. E.g. en-US
        
    """
    url = f"https://serp-api1.p.rapidapi.com/search"
    querystring = {'q': q, }
    if gl:
        querystring['gl'] = gl
    if cr:
        querystring['cr'] = cr
    if tbs:
        querystring['tbs'] = tbs
    if num:
        querystring['num'] = num
    if start:
        querystring['start'] = start
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serp-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

