import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_music_post_video_list(music_id: str, cursor: str='0', count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get music post video list"
    music_id: has more
        cursor: has more
        count: max 35 default 10
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getMusicVideoList"
    querystring = {'music_id': music_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
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
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getMusicInfo"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reply_list_by_commentid(comment_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get reply list by comment id"
    cursor: hasMore is True
        count: max 50
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getReplyListByCommentId"
    querystring = {'comment_id': comment_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
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
    url: 
https://www.tiktok.com/@tiktok/video/7093219391759764782
or
7093219391759764782
or
https://vm.tiktok.com/ZSeQS6B5k/
        count: max 50
        cursor: hasMore is True
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getCommentListByVideo"
    querystring = {'url': url, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def register_device_information(aid: str, version: str='250304', os: str='7.1.2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random device information,
		Activated"
    aid: 1180
1233
1340
        version: version code
        os: os version
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/RegisterDeviceInformation"
    querystring = {'aid': aid, }
    if version:
        querystring['version'] = version
    if os:
        querystring['os'] = os
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge_post_videos(challenge_id: str, count: str='10', region: str='US', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get challenge post videos"
    challenge_id: challenge_id (hashTag ID)
        count: max 35 default 10
        region: Get challenge videos for different regions
        cursor: hasMore is True.
load more
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getChallengeVideos"
    querystring = {'challenge_id': challenge_id, }
    if count:
        querystring['count'] = count
    if region:
        querystring['region'] = region
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_challenge(keywords: str, count: int=10, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search challenge"
    count: hasMore is True.
load more
        cursor: hasMore is True.
load more
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/searchChallenge"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
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
		input challenge_id or challenge_name"
    challenge_id: challenge_id or challenge_nane cannot be empty
        challenge_name: challenge_id or challenge_nane cannot be empty
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getChallengeInfo"
    querystring = {}
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
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
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getRegionList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
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
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getFeedVideoListByRegion"
    querystring = {'region': region, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video_list_by_keywords(keywords: str, count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search video list by keywords"
    count: max 30
        cursor: hasMore is true
load next page
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/searchVideoListByKeywords"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_play_list_by_userid(unique_id: str='@tiktok', cursor: str='0', count: str='10', user_id: str='107955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get playlist by user id
		unique_id or user_id is not empty"
    unique_id: unique_id
tiktok or @tiktok
        cursor: hasMore
        user_id: user_id
107955max 35
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getPlaylistByUserId"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_play_list_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get play list info"
    url: id or https://vm.tiktok.com/xxxxxxx
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getPlaylistInfo"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_play_list_post_video_list(count: str='10', cursor: str='0', mix_id: str='7163373594645482286', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get play list post video list"
    count: max 30 default 10
        cursor: has more
        mix_id: playlist id


        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getPlaylistVideoList"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if mix_id:
        querystring['mix_id'] = mix_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_list_by_useid(unique_id: str='tyler3497', user_id: str='6631770475491115014', cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection list by user id
		unique_id or user_id is not empty"
    unique_id: unique_id
tyler3497 or @tyler3497
        user_id: user_id
6631770475491115014
        cursor: hasMore
        count: max 35
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/collectionListByUserId"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_info(url: int=7214174961873849344, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection info"
    url: id or https://www.tiktok.com/@xxx/collection/xxx-xxxx
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/collectionInfo"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_post_video_list(collection_id: str, cursor: str='0', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get collection post video list"
    collection_id: collection id


        cursor: has more
        count: max 30 default 10
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getCollectionVideoList"
    querystring = {'collection_id': collection_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def et_ads_detail(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get creative center ads detail"
    url: or 7221117041168252930
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/biz/adDetail"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_favorite_videos(user_id: str='6741307595983946754', count: str='10', cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user favorite videos for latest
		unique_id or user_id is not empty"
    user_id: user_id
6529712362437328897
        count: max 35


        cursor: hasMore
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getUserFavoriteVideos"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(keywords: str, count: int=10, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user list by keywords"
    keywords: user nickname
        count: max 30
        cursor: cursor
hasMore is True, load next page
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/searchUser"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_userinfo(unique_id: str='@tiktok', user_id: int=107955, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user info
		unique_id or user_id is not empty"
    unique_id: user unique_id
tiktok or @tiktok
        user_id: user_id
107955
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getUserInfo"
    querystring = {}
    if unique_id:
        querystring['unique_id'] = unique_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_post_videos(user_id: str='107955', unique_id: str='@tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user post videos for latest
		get user feed
		unique_id or user_id is not empty"
    user_id: user_id
107955
        unique_id: unique_id
tiktok or @tiktok
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getUserVideos"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if unique_id:
        querystring['unique_id'] = unique_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_follower_list(user_id: int=107955, time: int=0, count: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user follower list"
    time: hasMore is True load next page
        count: max 200
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getUserfollowerList"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if time:
        querystring['time'] = time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following_list(user_id: int, count: int=50, time: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user following list"
    count: max 200
        time: hasMore is True load next page
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getUserFollowingList"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tiktok_video_info(url: str, hd: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
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
response: data.hdplay
        
    """
    url = f"https://tiktok-api15.p.rapidapi.com/index/Tiktok/getVideoInfo"
    querystring = {'url': url, }
    if hd:
        querystring['hd'] = hd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-api15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

