import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_threads(user: str, min_threads_length: str='3', store_in_airtable: str='false', use_checkpointing: str='false', since_date: str=None, since_tweet_id: str=None, limit: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of threads for the user specified in the url"
    
    """
    url = f"https://threads.p.rapidapi.com/threads/user/{user}"
    querystring = {}
    if min_threads_length:
        querystring['min_threads_length'] = min_threads_length
    if store_in_airtable:
        querystring['store_in_airtable'] = store_in_airtable
    if use_checkpointing:
        querystring['use_checkpointing'] = use_checkpointing
    if since_date:
        querystring['since_date'] = since_date
    if since_tweet_id:
        querystring['since_tweet_id'] = since_tweet_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "threads.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

