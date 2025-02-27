import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, related_keywords: str='true', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Web Search"
    query: Search query. You can use Google Search parameters.

e.g. `adele site:youtube.com`
e.g. `harry potter filetype:pdf`
e.g. `inurl:store`
e.g. `ronaldo -cristiano`
        related_keywords: Shows related keywords. Default: `false`

*Entering `true`, increases API latency*
        limit: Max results number. Max recommended value is `300`.
        
    """
    url = f"https://joj-web-search.p.rapidapi.com/"
    querystring = {'query': query, }
    if related_keywords:
        querystring['related_keywords'] = related_keywords
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-web-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

