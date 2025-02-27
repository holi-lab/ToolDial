import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def npm_search(query: str, page: int=None, ranking: str='popularity, quality, maintenance, optimal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search and query NPM packages"
    
    """
    url = f"https://npm-repository-search.p.rapidapi.com/"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if ranking:
        querystring['ranking'] = ranking
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "npm-repository-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

