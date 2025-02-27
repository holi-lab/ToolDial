import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_subreddit_names_memesare_and_freq(subreddit: str, freq: str, memesare: str, limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results by providing subreddit name, memesare and frequency (values are now, day, week, month, year, all) and limit should  have a valid value"
    
    """
    url = f"https://memes-from-reddit.p.rapidapi.com/{subreddit}/{memesare}/{freq}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memes-from-reddit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_subreddit_name_memesare(subreddit: str, memesare: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results by subreddit name and memesare (values are hot,new,top,rising) and limit should  have a valid value"
    
    """
    url = f"https://memes-from-reddit.p.rapidapi.com/{subreddit}/{memesare}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memes-from-reddit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_subreddit_name(subreddit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results by providing subreddit name (values must include meme,memes,joke,jokes)"
    
    """
    url = f"https://memes-from-reddit.p.rapidapi.com/{subreddit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memes-from-reddit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

