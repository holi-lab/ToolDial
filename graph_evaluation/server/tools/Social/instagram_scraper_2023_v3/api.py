import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_post_comments(media_pk: str, min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments made on a post. 
		
		Each post has a "pk" key.  To get the comments for a post, send the pk value of that post as media_pk in the url as query parameter"
    min_id: The value of next_min_id for viewing the current comments list
        
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/post_comments"
    querystring = {'media_pk': media_pk, }
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts(handle: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user posts. Max - 33
		Use end_cursor to get the rest of the data"
    max_id: The value of next_max_id for viewing the current posts
        
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/posts"
    querystring = {'handle': handle, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user info by username"
    
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/userinfo"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers(user_id: str, max_id: str=None, count: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user followers. Max - 100"
    max_id: The value of next_max_id for viewing the followers list
        
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/followers"
    querystring = {'user_id': user_id, }
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following(user_id: str, max_id: str=None, count: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user following. Max - 100"
    count: The value of next_max_id for viewing the following list
        
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/following"
    querystring = {'user_id': user_id, }
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_id(handle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user id"
    
    """
    url = f"https://instagram-scraper-2023.p.rapidapi.com/userid"
    querystring = {'handle': handle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

