import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_s_detail(user_id: str='107955', unique_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User's Detail, **unique_id** or **user_id** is not empty"
    
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/info"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_favorite_videos(cursor: str='0', user_id: str='6741307595983946754', unique_id: str=None, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User's Favorite Videos"
    cursor: Used to get the content of the next page
        count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/favorite"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_followings(user_id: str, time: str='0', count: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User's Followings"
    user_id: user's id
        time: Used to get the content of the next page
        count: `count` has a maximum value of 200
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/following"
    querystring = {'user_id': user_id, }
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_post(cursor: str='0', count: str='10', unique_id: str=None, user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User's Videos"
    cursor: Used to get the content of the next page
        count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/posts"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video(keywords: str, sort_type: int=0, publish_time: int=0, cursor: str='0', region: str='us', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Video By Keywords"
    sort_type: Sort by

0 - Relevance
1 - Like count
3 - Date posted
        publish_time: Publish time filter

0 - ALL
1 - Past 24 hours
7 - This week
30 - This month
90 - Last 3 months
180 - Last 6 months
        region: region code, such as: **jp, kr, us, vn, br, ru** ...
        count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/feed/search"
    querystring = {'keywords': keywords, }
    if sort_type:
        querystring['sort_type'] = sort_type
    if publish_time:
        querystring['publish_time'] = publish_time
    if cursor:
        querystring['cursor'] = cursor
    if region:
        querystring['region'] = region
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_s_videos(music_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music's Videos"
    count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/music/posts"
    querystring = {'music_id': music_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_s_detail(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music's Detail"
    url: music link or id
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/music/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_challenge(keywords: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Challenge"
    count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/challenge/search"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_s_videos(challenge_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Challenge's Videos"
    challenge_id: challenge id (hashtag id)
        count: max 30
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/challenge/posts"
    querystring = {'challenge_id': challenge_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_s_detail(challenge_name: str=None, challenge_id: str='33380', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Challenge's Detail"
    challenge_name: challenge name
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/challenge/info"
    querystring = {}
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comment_reply(comment_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Comment Reply"
    
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/comment/reply"
    querystring = {'comment_id': comment_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_s_comments(url: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Video's Comments"
    url: This can be a link or id
        count: max 50
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/comment/list"
    querystring = {'url': url, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_videos(region: str, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Videos By Region"
    region: region code, such as: **jp, kr, us, vn, br, ru** ...
        count: max 20
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/feed/list"
    querystring = {'region': region, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(count: str='10', cursor: str='0', keywords: str='tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Users"
    count: max 30
        cursor: Used to get the content of the next page
        keywords: user's nickname
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/search"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_followers(user_id: str, time: str='0', count: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User's Followers"
    user_id: user's id
        time: Used to get the content of the next page
        count: `count` has a maximum value of 200
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/user/followers"
    querystring = {'user_id': user_id, }
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_device(aid: str, os: str='7.1.2', version: str='250304', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate device, activated"
    
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/service/registerDevice"
    querystring = {'aid': aid, }
    if os:
        querystring['os'] = os
    if version:
        querystring['version'] = version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_detail(url: str, hd: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Video Detail, High quality
		Support Douyin"
    hd: Get Original Quality
        
    """
    url = f"https://tiktok-scraper7.p.rapidapi.com/"
    querystring = {'url': url, }
    if hd:
        querystring['hd'] = hd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

