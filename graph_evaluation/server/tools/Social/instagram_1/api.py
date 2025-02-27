import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hashtag(hashtag: str, raw: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Posts by hashtag"
    raw: set to 'true' or '1' to return the raw unprocessed data feed
        
    """
    url = f"https://instagram-1.p.rapidapi.com/hashtag/{hashtag}"
    querystring = {}
    if raw:
        querystring['raw'] = raw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feed_by_username(username: str, raw: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Collect the current feed from the username"
    raw: set to 'true' or '1' to return the raw unprocessed data feed
        
    """
    url = f"https://instagram-1.p.rapidapi.com/feed/{username}"
    querystring = {}
    if raw:
        querystring['raw'] = raw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

