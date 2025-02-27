import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def posts(creator_id: int, cursor: str=None, amount: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts of a Creator with the `creator_id`. The `cursor` to the next page will be returned with this request."
    creator_id: The `creator_id`can be obtained using the `/search` endpoint.
        cursor: The cursor to the next page of results.
        amount: The `amount`of posts to be returned. Default is `10`. Max is `500`. Bigger amount results in higher response time.
        
    """
    url = f"https://patreon8.p.rapidapi.com/posts/{creator_id}"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patreon8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details(creator_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details and general information of a Creator with the `creator_id`."
    creator_id: The `creator_id` can be obtained using the `/search` endpoint.
        
    """
    url = f"https://patreon8.p.rapidapi.com/details/{creator_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patreon8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_creators(search_query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Creators on Patreon using the `search_query`."
    search_query: Search term used in Patreon search.
        
    """
    url = f"https://patreon8.p.rapidapi.com/search/{search_query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patreon8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

