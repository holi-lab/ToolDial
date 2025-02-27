import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_comment_list_by_video(url: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get comment list by video"
    url: https://www.tiktok.com/@tiktok/video/7093219391759764782
or
7093219391759764782
        count: max 50
        cursor: hasMore is True
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/commentList"
    querystring = {'url': url, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reply_list_by_comment_id(comment_id: str, count: str='10', cursor: str='o', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get reply list by comment id"
    count: max 50
        cursor: hasMore is True
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/commentReply"
    querystring = {'comment_id': comment_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_region_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get region list"
    
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/regionList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_feed_video_list_by_region(region: str, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get feed video list by region"
    region: region code
by get region list api
        count: max 20
Inaccurate


        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/feedList"
    querystring = {'region': region, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video_list_by_keywords(keywords: str, sort_type: str='0', publish_time: str='0', cursor: str='0', region: str='US', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search video list by keywords"
    sort_type: 
Sort by

0 - Relevance
1 - Like count
3 - Date posted
        publish_time: 
Publish time filter

0 - ALL
1 - Past 24 hours
7 - This week
30 - This month
90 - Last 3 months
180 - Last 6 months
        cursor: Search for videos from different regions
        count: hasMore is true
load next page
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/feedSearch"
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
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tiktok_video_info(hd: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tiktok video full info. HD Quality, No Watermark. Fast.
		Support Tiktok & Douyin.
		Support Getting Image List.
		Support Tiktok Stories."
    hd: Get HD Video(High bit rate). This increases the total request time a little.
response: data.hdplay
        url: 7106658991907802411
or
https://www.tiktok.com/@tiktok/video/7106658991907802411
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/getVideo"
    querystring = {'hd': hd, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following_list(user_id: str, count: str, time: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user following list"
    count: max 200
        time: 
hasMore is True load next page
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/userFollowingList"
    querystring = {'user_id': user_id, 'count': count, 'time': time, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(cursor: str, keywords: str, count: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user list by keywords"
    cursor: OPTIONAL
cursor
hasMore is True, load next page
        keywords: REQUIRED
user nickname
        count: max 30
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/searchUser"
    querystring = {'cursor': cursor, 'keywords': keywords, 'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get music info"
    url: 
id or https://vm.tiktok.com/xxxxxxx
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/musicInfo"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_post_video_list(cursor: str, count: str, music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get music post video list"
    cursor: has more
        count: max 35 default 10
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/musicVideo"
    querystring = {'cursor': cursor, 'count': count, 'music_id': music_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_follower_list(user_id: str, time: str='0', count: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user follower list"
    time: OPTIONAL
hasMore is True load next page
        count: max 200
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/userFollowerList"
    querystring = {'user_id': user_id, }
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info(user_id: str='107955', unique_id: str='@tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user info
		unique_id or user_id is not empty"
    
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/userInfo"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_post_videos(count: str='10', unique_id: str='@tiktok', cursor: str='0', user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user post videos for latest
		get user feed
		unique_id or user_id is not empty"
    count: max 35
        unique_id: unique_id
tiktok or @tiktok
        cursor: hasMore
        user_id: user_id
107955
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/userPublishVideo"
    querystring = {}
    if count:
        querystring['count'] = count
    if unique_id:
        querystring['unique_id'] = unique_id
    if cursor:
        querystring['cursor'] = cursor
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_favorite_videos(unique_id: str='@siicantikkk15', count: str='10', cursor: str='0', user_id: str='6741307595983946754', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user favorite videos for latest
		unique_id or user_id is not empty"
    unique_id: unique_id
mineny13 or @mineny13
        count: max 35
        cursor: hasMore
        user_id: 6741307595983946754
        
    """
    url = f"https://tiktok-download-video1.p.rapidapi.com/userFavoriteVideo"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

