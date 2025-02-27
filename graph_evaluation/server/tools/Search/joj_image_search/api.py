import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(keyword: str, max: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image Search"
    
    """
    url = f"https://joj-image-search.p.rapidapi.com/v1/"
    querystring = {'keyword': keyword, 'max': max, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_v2_recommended(q: str, nfpr: str=None, tbs: str=None, safe: str=None, filter: str=None, suggestion_query: str=None, cursor: str=None, hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image Search (V2)"
    q: Search query. You can use search parameters.

e.g. `adele site:youtube.com`
e.g. `query -exclude`
        tbs: Search filters.
        safe: If you want safe search, set it to `active`
        suggestion_query: Suggestion query (must be taken from response suggestions).
        cursor: Cursor for next results.
        hl: Language code for search. Default: en
        
    """
    url = f"https://joj-image-search.p.rapidapi.com/v2/"
    querystring = {'q': q, }
    if nfpr:
        querystring['nfpr'] = nfpr
    if tbs:
        querystring['tbs'] = tbs
    if safe:
        querystring['safe'] = safe
    if filter:
        querystring['filter'] = filter
    if suggestion_query:
        querystring['suggestion_query'] = suggestion_query
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

