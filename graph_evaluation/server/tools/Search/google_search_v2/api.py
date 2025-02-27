import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, limit: int=10, related_keywords: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    query: Search query. You can use Google Search parameters.

e.g. adele site:youtube.com
e.g. harry potter filetype:pdf
e.g. inurl:store
e.g. ronaldo -cristiano
        limit: Max results number. Max recommended value is 300
        related_keywords: Shows related keywords. Default: false

Entering true, increases API latency
        
    """
    url = f"https://google-search74.p.rapidapi.com/"
    querystring = {'query': query, }
    if limit:
        querystring['limit'] = limit
    if related_keywords:
        querystring['related_keywords'] = related_keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

