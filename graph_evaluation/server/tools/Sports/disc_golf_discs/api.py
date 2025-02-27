import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_disc_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a certain discs when given an ID"
    
    """
    url = f"https://disc-golf-discs.p.rapidapi.com/discs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disc-golf-discs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_discs(limit: int=50, name: str=None, namelike: str=None, offset: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the discs golf discs on the market"
    
    """
    url = f"https://disc-golf-discs.p.rapidapi.com/discs"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if namelike:
        querystring['nameLike'] = namelike
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disc-golf-discs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

