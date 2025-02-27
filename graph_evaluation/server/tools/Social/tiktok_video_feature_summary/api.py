import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_challenge_post_videos(challenge_id: str, count: str='10', cursor: str='0', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get challenge post videos"
    challenge_id: challenge_id (hashTag ID)
        count: max 20 default 10
        cursor: hasMore is True 
which value in prev response to load more
        region: Get challenge videos for different regions
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/challenge/posts"
    querystring = {'challenge_id': challenge_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_info(challenge_name: str='cosplay', challenge_id: str='33380', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get challenge detail
		input challenge_id or challenge_name"
    challenge_name: challenge_id or challenge_nane cannot be empty
        challenge_id: challenge_id or challenge_nane cannot be empty
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/challenge/info"
    querystring = {}
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_challenge(keywords: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search challenge"
    cursor: hasMore is True.
load more
        count: max 30 default 10
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/challenge/search"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_list_by_user_id(cursor: str='0', unique_id: str='tyler3497', count: str='10', user_id: str='6631770475491115014', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection list by user id
		unique_id or user_id is not empty"
    cursor: hasMore
        unique_id: unique_id
tyler3497 or @tyler3497
        count: max 35
        user_id: user_id
6631770475491115014
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/collection/list"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if unique_id:
        querystring['unique_id'] = unique_id
    if count:
        querystring['count'] = count
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection info"
    url: id or https://www.tiktok.com/@xxx/collection/xxx-xxxx
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/collection/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_post_video_list(collection_id: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection post video list"
    collection_id: collection_id
        count: max 30 default 10
        cursor: has more
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/collection/posts"
    querystring = {'collection_id': collection_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_by_user_id(unique_id: str='@tiktok', user_id: str='107955', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist by user id
		unique_id or user_id is not empty"
    unique_id: unique_id
tiktok or @tiktok
        user_id: user_id
107955
        count: max 35
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/mix/list"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist info"
    url: id or https://vm.tiktok.com/xxxxxxx
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/mix/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_post_video_list(mix_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist post video list"
    mix_id: playlist id
        cursor: has more
        count: max 30 default 10
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/mix/posts"
    querystring = {'mix_id': mix_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tiktok_video_full_info(url: str, hd: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Support Tiktok & Douyin.
		Returns relevant information about querying video addresses, 
		including high-definition watermark free video addresses, 
		author information, 
		background music information, 
		views, 
		likes, 
		comments, 
		etc- List Item"
    url: Tiktok or Douyin video address
        hd: Get HD Video(High bit rate). This increases the total request time a little.
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/"
    querystring = {'url': url, }
    if hd:
        querystring['hd'] = hd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_list_by_video(url: str, count: int=10, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get comment list by video"
    url: https://www.tiktok.com/@tiktok/video/7093219391759764782
or
7093219391759764782
or
https://vm.tiktok.com/ZSeQS6B5k/
        count: max 50
        cursor: hasMore is True
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/comment/list"
    querystring = {'url': url, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def register_device_information(aid: int, version: str='250304', os: str='7.1.2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random device information,
		Activated"
    aid: 1180
1233
1340
        version: version code
        os: os version
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/service/registerDevice"
    querystring = {'aid': aid, }
    if version:
        querystring['version'] = version
    if os:
        querystring['os'] = os
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get music info base on id"
    url: id or https://vm.tiktok.com/xxxxxxx
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/music/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following_list(user_id: str, count: str='50', time: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user following list"
    count: max 200
        time: hasMore is True load next page
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/following"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_follower_list(user_id: str, count: str='50', time: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user follower list"
    count: max 200
        time: hasMore is True load next page
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/followers"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_video_with_keywords(keywords: str, count: str='10', publish_time: int=0, cursor: str='0', region: str='US', sort_type: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get related video  list with list"
    publish_time: 0 - ALL
1 - Past 24 hours
7 - This week
30 - This month
90 - Last 3 months
180 - Last 6 months
        region: Please refer to the region list interface for details
        sort_type: 0 - Relevance
1 - Like count
3 - Date posted
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/feed/search"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if publish_time:
        querystring['publish_time'] = publish_time
    if cursor:
        querystring['cursor'] = cursor
    if region:
        querystring['region'] = region
    if sort_type:
        querystring['sort_type'] = sort_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def region_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the region list use in video search params"
    
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/region"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_detail_info(unique_id: str='voyagel', user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get users detail info
		unique_id or user_id is not empty"
    
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/info"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_favorite_videos(cursor: str='0', user_id: str='6741307595983946754', unique_id: str='voyagel', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user favorite videos list"
    unique_id: voyagel or @voyagel
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/favorite"
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
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_video(cursor: str='0', unique_id: str=None, user_id: str='107955', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user post videos
		get user feed
		unique_id or user_id is not empty"
    unique_id: unique_id
tiktok or @tiktok
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/posts"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users_data(keywords: str, cursor: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get users data list by keywords"
    keywords: users nickname
Support for fuzzy search
        cursor: hasMore is True, load next page
        count: return count
        
    """
    url = f"https://tiktok-video-feature-summary.p.rapidapi.com/user/search"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-feature-summary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

