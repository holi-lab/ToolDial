import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get real-time organic search results from across the web. Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc)."
    q: Search query.  Supports all Google advanced search operators (*site:*, *inurl:*, *intitle:*, etc).

e.g. `website builder`
e.g. ` site:youtube.com`
e.g. `nda filetype:pdf`
e.g. `cristiano -ronaldo`
        limit: Maximum number of results to return (1-300). 
**Default**: `10`.
        
    """
    url = f"https://real-time-web-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

