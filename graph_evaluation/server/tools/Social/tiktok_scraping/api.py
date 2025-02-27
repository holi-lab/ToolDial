import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_comments_reply(aweme_id: str, comment_id: str, cursor: int=0, count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "list comments reply endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/list-comments-reply"
    querystring = {'aweme_id': aweme_id, 'comment_id': comment_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comments_of_post(aweme_id: str, cursor: int=None, count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "list comments of post endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/list-comments"
    querystring = {'aweme_id': aweme_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likes_of_user(max_cursor: int=0, count: int=20, user_id: str='6546356850533602319', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user likes endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/user-likes"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if count:
        querystring['count'] = count
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_of_user(max_cursor: int=0, count: int=20, sec_user_id: str='MS4wLjABAAAAxmiNgjKTNw0rn5WwweMsKe4WzgvbxRT_xeKhn-bA4O4HM3VS8piCMo88EXopqKh7', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get posts of user endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/user-posts"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if count:
        querystring['count'] = count
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_sounds(keyword: str, cursor: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search sounds endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/search-sounds"
    querystring = {'keyword': keyword, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_of_hashtag(cid: str, count: int=20, cursor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get posts of hashtag endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/hashtag-posts"
    querystring = {'cid': cid, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_posts(keyword: str, offset: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search posts endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/search-posts"
    querystring = {'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_of_sound(music_id: str, count: int=20, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get posts of sound endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/music-posts"
    querystring = {'music_id': music_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(keyword: str, count: int=None, cursor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search users endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/search-users"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags(keyword: str, cursor: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search hashtags endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/search-hashtags"
    querystring = {'keyword': keyword, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sound(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get sounds endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/get-music"
    querystring = {'music_id': music_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers(count: int=20, max_time: int=0, user_id: str=None, sec_user_id: str='MS4wLjABAAAAxmiNgjKTNw0rn5WwweMsKe4WzgvbxRT_xeKhn-bA4O4HM3VS8piCMo88EXopqKh7', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user followers endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/list-followers"
    querystring = {}
    if count:
        querystring['count'] = count
    if max_time:
        querystring['max_time'] = max_time
    if user_id:
        querystring['user_id'] = user_id
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_general(keyword: str, offset: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search general endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following(max_time: int=0, count: int=20, sec_user_id: str='MS4wLjABAAAAxmiNgjKTNw0rn5WwweMsKe4WzgvbxRT_xeKhn-bA4O4HM3VS8piCMo88EXopqKh7', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user following endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/list-following"
    querystring = {}
    if max_time:
        querystring['max_time'] = max_time
    if count:
        querystring['count'] = count
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post(aweme_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get post endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/get-post"
    querystring = {'aweme_id': aweme_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(user_id: str=None, sec_user_id: str='MS4wLjABAAAAxmiNgjKTNw0rn5WwweMsKe4WzgvbxRT_xeKhn-bA4O4HM3VS8piCMo88EXopqKh7', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user endpoint"
    
    """
    url = f"https://tiktok-scraping.p.rapidapi.com/get-user"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

