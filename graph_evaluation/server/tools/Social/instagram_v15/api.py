import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_feed_by_username(by: str, username: str, pageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user feed by `username`, you can get the next page by providing `pageId` query string with `next_page` from the response."
    
    """
    url = f"https://instagram98.p.rapidapi.com/account/{username}/feed"
    querystring = {'by': by, }
    if pageid:
        querystring['pageId'] = pageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram98.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you will able to fetch user profile data using `username`, this includes user profile info and stats like media, followers and following counts, business data and more.
		
		Query Parameters:
		
		`by` indicates input type for finding the user can be `id` or `username` it's username by default"
    
    """
    url = f"https://instagram98.p.rapidapi.com/account/{username}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram98.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stories(user_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you will able to fetch user stories using `user_id`. you can even provide multiple ids to fetch more stories with single API call, simple separate them by comma: `user_id_1`, `user_id_2`."
    
    """
    url = f"https://instagram98.p.rapidapi.com/account/{user_ids}/stories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram98.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_by_id(user_id: str, by: str='id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this endpoint you will able to fetch user profile data using `id`, this includes user profile info and stats like media, followers and following counts, business data and more.
		
		Query Parameters:
		
		- `by` indicates input type for finding the user can be id or username it's username by default"
    
    """
    url = f"https://instagram98.p.rapidapi.com/account/{user_id}/info"
    querystring = {}
    if by:
        querystring['by'] = by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram98.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

