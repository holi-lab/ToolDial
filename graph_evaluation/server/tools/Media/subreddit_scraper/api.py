import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def subreddit(q: str, limit: str='25', after: str=None, before: str=None, time: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subreddit"
    
    """
    url = f"https://subreddit-scraper.p.rapidapi.com/subreddit"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if after:
        querystring['after'] = after
    if before:
        querystring['before'] = before
    if time:
        querystring['time'] = time
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subreddit-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

