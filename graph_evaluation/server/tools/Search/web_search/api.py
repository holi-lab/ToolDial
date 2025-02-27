import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web_search(query: str, related_keywords: str=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Web Search Results"
    query: Search query. You can use web search parameters.

e.g. `ronaldo site:instagram.com`
e.g. `twitter filetype:pdf`
e.g. `inurl:market`
        related_keywords: Shows related keywords. Default: `false`

Entering `true`, increases API latency
        limit: Max results number. Max recommended value is `300`.
        
    """
    url = f"https://web-search24.p.rapidapi.com/"
    querystring = {'query': query, }
    if related_keywords:
        querystring['related_keywords'] = related_keywords
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

