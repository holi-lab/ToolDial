import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(term: str, size: int=64, limit: int=20, color: str='ff0000', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Icons8 repository"
    term: Search term
        size: Size of icon in px
        color: Color of icon
        
    """
    url = f"https://unofficial-icons8-search.p.rapidapi.com/search"
    querystring = {'term': term, }
    if size:
        querystring['size'] = size
    if limit:
        querystring['limit'] = limit
    if color:
        querystring['color'] = color
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-icons8-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

