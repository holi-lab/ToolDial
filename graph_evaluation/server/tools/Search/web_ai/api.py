import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, freshness: str=None, cc: str=None, safesearch: str=None, setlang: str=None, offset: str='0', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Web pages search results, related searches and query context"
    q: search query terms
        freshness: Day
Week
Month
        cc: country code
        safesearch: Off
Moderate
Strict
        setlang: language code
        offset: skip this many results
        count: number of results
≤ 50
        
    """
    url = f"https://web-ai.p.rapidapi.com/search"
    querystring = {'q': q, }
    if freshness:
        querystring['freshness'] = freshness
    if cc:
        querystring['cc'] = cc
    if safesearch:
        querystring['safeSearch'] = safesearch
    if setlang:
        querystring['setLang'] = setlang
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

