import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_image_search(q: str, suggestion_query: str=None, hl: str='en', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / Image Search"
    q: Search Query
        suggestion_query: Suggestion Query Token
        hl: Language
        cursor: Cursor Token
        
    """
    url = f"https://image-search19.p.rapidapi.com/v2/"
    querystring = {'q': q, }
    if suggestion_query:
        querystring['suggestion_query'] = suggestion_query
    if hl:
        querystring['hl'] = hl
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-search19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_image_search(q: str, limit: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1 / Image Search"
    q: Search Query
        
    """
    url = f"https://image-search19.p.rapidapi.com/v1/"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-search19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

