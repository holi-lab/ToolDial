import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_top_5_latest_posts_from_reddit(after: str=None, sorting: str=None, subreddit: str=None, timeframe: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the top 5 posts, uploaded today. These are sorted by upvotes."
    
    """
    url = f"https://programmer-humor.p.rapidapi.com/api/reddit"
    querystring = {}
    if after:
        querystring['after'] = after
    if sorting:
        querystring['sorting'] = sorting
    if subreddit:
        querystring['subreddit'] = subreddit
    if timeframe:
        querystring['timeframe'] = timeframe
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "programmer-humor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_10_posts_from_9gag(after: int=5, sorting: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looks for any of these tags
		'programming', 'coding', 'code', 'tech', 'developer', 'webdev', 'web-dev'"
    
    """
    url = f"https://programmer-humor.p.rapidapi.com/api/9gag"
    querystring = {}
    if after:
        querystring['after'] = after
    if sorting:
        querystring['sorting'] = sorting
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "programmer-humor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

