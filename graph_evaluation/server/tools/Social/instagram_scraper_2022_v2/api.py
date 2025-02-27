import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_profile_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user information  by username"
    
    """
    url = f"https://instagram-scraper-20221.p.rapidapi.com/userinfo"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
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
    url = f"https://instagram-scraper-20221.p.rapidapi.com/userid"
    querystring = {'handle': handle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
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
    
    """
    url = f"https://instagram-scraper-20221.p.rapidapi.com/posts"
    querystring = {'handle': handle, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers(user_id: str, count: str='100', max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user followers. Max - 100"
    
    """
    url = f"https://instagram-scraper-20221.p.rapidapi.com/followers"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following(user_id: str, count: str='100', max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user following. Max - 100"
    
    """
    url = f"https://instagram-scraper-20221.p.rapidapi.com/following"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post_comments(media_pk: str, min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments made on a post. 
		
		Each post has a "pk" key.  To get the comments for a post, send the pk value of that post as media_pk in the url as query parameter"
    
    """
    url = f"https://instagram-scraper-20221.p.rapidapi.com/post_comments"
    querystring = {'media_pk': media_pk, }
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-20221.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

