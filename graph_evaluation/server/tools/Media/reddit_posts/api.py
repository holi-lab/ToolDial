import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_posts_top(ordering: str, subreddit: str, start: int, time: str='year', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View information on posts, query by subreddit, sort, and location in list. This request demonstrates how to request top posts using the time parameter (which must be chosen from the following options: hour, day, week, month, year, all)."
    
    """
    url = f"https://reddit-posts.p.rapidapi.com/get-posts"
    querystring = {'ordering': ordering, 'subreddit': subreddit, 'start': start, }
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-posts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_posts_default(start: int, subreddit: str, ordering: str, time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View information on posts, query by subreddit, sort, and location in list. This request demonstrates the default response of the reddit homepage."
    
    """
    url = f"https://reddit-posts.p.rapidapi.com/get-posts"
    querystring = {'start': start, 'subreddit': subreddit, 'ordering': ordering, }
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-posts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

