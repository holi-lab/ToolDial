import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_no_watermark(video_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video no watermark."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/video/no_watermark"
    querystring = {'video_url': video_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_videos(sec_uid: str='MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM', user_id: str='107955', max_cursor: str=None, user_name: str='tiktok', min_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user videos."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/user/videos"
    querystring = {}
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if user_id:
        querystring['user_id'] = user_id
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if user_name:
        querystring['user_name'] = user_name
    if min_cursor:
        querystring['min_cursor'] = min_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info(user_id: str='107955', user_name: str='tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic user information."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/user/info"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if user_name:
        querystring['user_name'] = user_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_follower(user_id: str, count: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers."
    cursor: min_time from response, default 0
        
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/user/follower"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followings(user_id: str, count: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followings."
    cursor: min_time from response, default 0
        
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/user/following"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(video_url: str, count: int=None, cursor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video comments."
    cursor: min_time from response, default 0
        
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/video/comments"
    querystring = {'video_url': video_url, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_videos(music_id: str, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos with specific music."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/music/videos"
    querystring = {'music_id': music_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_info(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music info."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/music/info"
    querystring = {'music_id': music_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_videos(hashtag_id: str, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag (challenge) videos."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/hashtag/videos"
    querystring = {'hashtag_id': hashtag_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_info(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag (challenge) info."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/hashtag/info"
    querystring = {'hashtag': hashtag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_info(video_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video info."
    
    """
    url = f"https://tiktok-scraper2.p.rapidapi.com/video/info"
    querystring = {'video_url': video_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

