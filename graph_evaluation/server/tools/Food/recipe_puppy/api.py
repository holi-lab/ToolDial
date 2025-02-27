import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str='omelet', p: str='1', i: str='onions,garlic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Recipe Puppy database"
    q: Search Query
        p: page
        i: Comma separated ingredients
        
    """
    url = f"https://recipe-puppy.p.rapidapi.com/"
    querystring = {}
    if q:
        querystring['q'] = q
    if p:
        querystring['p'] = p
    if i:
        querystring['i'] = i
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-puppy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

