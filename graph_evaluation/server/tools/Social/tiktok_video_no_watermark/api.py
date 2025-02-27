import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_similar_users(user_id: str='6821796598806348805', unique_id: str='ovaksss', count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### **Notice**! ! !  This endpoint is deprecated! ! !
		
		Discover other users similar to the target user."
    user_id: user_id
6821796598806348805
        unique_id: user unique_id
ovaksss or    @ovaksss
        count: max 50
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/discover"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def increase_download_count(is_id: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Increase download count.
		Support Tiktok Video, Story ...
		**There is a delay of 3~5 seconds for statistics counting**"
    id: Video's id
        region: Default random region viewer.
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/smm/download"
    querystring = {'id': is_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def increase_share_count(is_id: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Increase share count.
		Support Tiktok Video, Story ...
		**There is a delay of 3~5 seconds for statistics counting**"
    id: video id
        region: Default random region viewer.
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/smm/share"
    querystring = {'id': is_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def increase_views(is_id: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Increase views(play_count).
		Support Tiktok Video, Story ...
		**There is a delay of 3~5 seconds for statistics counting**"
    id: Video's id
        region: Default random region viewer.
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/smm/views"
    querystring = {'id': is_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def register_device_information(aid: int, os: str='7.1.2', version: str='250304', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random device information,
		Activated"
    aid: 1180
1233
1340
        os: os version
        version: version code
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/service/registerDevice"
    querystring = {'aid': aid, }
    if os:
        querystring['os'] = os
    if version:
        querystring['version'] = version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_post_videos(challenge_id: str, count: str='10', cursor: str='0', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get challenge post videos"
    challenge_id: challenge_id (hashTag ID)
        count: max 35 default 10
        cursor: hasMore is True. 
load more
        region: Get challenge videos for different regions
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/challenge/posts"
    querystring = {'challenge_id': challenge_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video_list_by_keywords(keywords: str, count: str='10', region: str='US', publish_time: int=0, sort_type: int=0, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search video list by keywords"
    count: max 30
        region: Search for videos from different regions
        publish_time: Publish time filter

0 - ALL
1 - Past 24 hours
7 - This week
30 - This month
90 - Last 3 months
180 - Last 6 months
        sort_type: Sort by

0 - Relevance
1 - Like count
3 - Date posted
        cursor: hasMore is true
load next page
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/feed/search"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if region:
        querystring['region'] = region
    if publish_time:
        querystring['publish_time'] = publish_time
    if sort_type:
        querystring['sort_type'] = sort_type
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tiktok_video_info(url: str, hd: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tiktok video full info. HD Quality, No Watermark. Fast.
		Support Tiktok & Douyin.
		Support Getting Image List.
		Support Tiktok Stories."
    url: https://vt.tiktok.com/ZSdGG1Y1k/
or
https://www.tiktok.com/@tiktok/video/7106658991907802411
or
7106658991907802411

Image list
https://vm.tiktok.com/ZMNkqKUce/
        hd: Get HD Video(High bit rate). This increases the total request time a little.
response:   data.hdplay
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/"
    querystring = {'url': url, }
    if hd:
        querystring['hd'] = hd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    collection_id: collection id
        count: max 30 default 10
        cursor: has more
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/collection/posts"
    querystring = {'collection_id': collection_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_post_video_list(mix_id: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist post video list"
    mix_id: playlist id
        count: max 30 default 10
        cursor: has more
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/mix/posts"
    querystring = {'mix_id': mix_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/collection/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_list_by_user_id(unique_id: str='tyler3497', user_id: str='6631770475491115014', count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection list by user id
		unique_id or user_id is not empty"
    unique_id: unique_id
tyler3497 or @tyler3497
        user_id: user_id
6631770475491115014
        count: max 35
        cursor: hasMore
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/collection/list"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/mix/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_by_user_id(count: str='10', user_id: str='107955', unique_id: str='@tiktok', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist by user id
		unique_id or user_id is not empty"
    count: max 35
        user_id: user_id
107955
        unique_id: unique_id
tiktok or @tiktok
        cursor: hasMore
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/mix/list"
    querystring = {}
    if count:
        querystring['count'] = count
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info(unique_id: str='@tiktok', user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user info
		unique_id or user_id is not empty"
    unique_id: user unique_id
tiktok or    @tiktok
        user_id: user_id
107955
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/info"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(keywords: str, cursor: str='0', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user list by keywords"
    keywords: user nickname
        cursor: cursor
hasMore is True, load next page
        count: max 30
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/search"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_favorite_videos(count: str='10', cursor: str='0', unique_id: str='@siicantikkk15', user_id: str='6741307595983946754', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user favorite videos for latest
		unique_id or user_id is not empty"
    count: max 35
        cursor: hasMore
        unique_id: unique_id
mineny13 or @mineny13
        user_id: user_id
6529712362437328897
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/favorite"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_post_videos(unique_id: str='@tiktok', count: str='10', cursor: str='0', user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user post  videos for latest
		get user feed
		unique_id or user_id is not empty"
    unique_id: unique_id
tiktok or @tiktok
        count: max 35
        cursor: hasMore
        user_id: user_id
107955
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/posts"
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
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reply_list_by_comment_id(comment_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get reply list by comment id"
    cursor: hasMore is True
        count: max 50
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/comment/reply"
    querystring = {'comment_id': comment_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comment_list_by_video(url: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
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
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/comment/list"
    querystring = {'url': url, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_follower_list(user_id: str, time: str='0', count: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user follower list"
    time: hasMore is True  load next page
        count: max 200
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/followers"
    querystring = {'user_id': user_id, }
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following_list(user_id: str, time: str='0', count: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user following list"
    time: hasMore is True  load next page
        count: max 200
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/user/following"
    querystring = {'user_id': user_id, }
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/feed/list"
    querystring = {'region': region, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_challenge(keywords: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search challenge"
    count: max 30 default 10
        cursor: hasMore is True. 
load more
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/challenge/search"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_info(challenge_id: str='33380', challenge_name: str='cosplay', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get challenge detail
		input  challenge_id or challenge_name"
    challenge_id: challenge_id or challenge_nane cannot be empty
        challenge_name: challenge_id or challenge_nane cannot be empty
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/challenge/info"
    querystring = {}
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/region"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_post_video_list(music_id: str, count: str='20', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get music post video list"
    count: max 35 default 10
        cursor: has more
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/music/posts"
    querystring = {'music_id': music_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
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
    url: id or https://vm.tiktok.com/xxxxxxx
        
    """
    url = f"https://tiktok-video-no-watermark2.p.rapidapi.com/music/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

