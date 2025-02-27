import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detail(is_id: int, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve film details"
    
    """
    url = f"https://filmaffinity-unofficial.p.rapidapi.com/movie/detail/"
    querystring = {'id': is_id, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "filmaffinity-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, lang: str, toyear: int=None, fromyear: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search films by title - returns max 100 movies"
    
    """
    url = f"https://filmaffinity-unofficial.p.rapidapi.com/movie/search/"
    querystring = {'query': query, 'lang': lang, }
    if toyear:
        querystring['toYear'] = toyear
    if fromyear:
        querystring['fromYear'] = fromyear
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "filmaffinity-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

