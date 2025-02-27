import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(q: str, offset: int=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news stories by keyword(s)"
    q: Query Term(s)
        offset: Offset of first item in result (default 0)
        count: Number of items in response (default 10, max 100)
        
    """
    url = f"https://newslit-news-search.p.rapidapi.com/news"
    querystring = {'q': q, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newslit-news-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

