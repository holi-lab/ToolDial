import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_music_post_video(music_id: int, cursor: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music post video list"
    cursor: next page
        count: max=30
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getmusicpostvideolist"
    querystring = {'music_id': music_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_post_video(user_id: int, min_cursor: str='0', max_cursor: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user post video"
    
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getpostuser"
    querystring = {'user_id': user_id, }
    if min_cursor:
        querystring['min_cursor'] = min_cursor
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_challege(type: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info challege"
    type: challengeId or challengeName
        value: id challenge or name challenge
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getchallengeinfo"
    querystring = {'type': type, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_music(keywords: str, count: int=5, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search music"
    count: max=30
        cursor: next page
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/searchmusic"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_challenge(keywords: str, cursor: int=0, count: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search challenge by keywords"
    cursor: next page
        count: max=5
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/searchchallenge"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reply_comment_by_id(cid: int, aweme_id: int, count: int=5, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reply comment by id"
    count: max=30
        cursor: next page
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getcommentreplylist"
    querystring = {'cid': cid, 'aweme_id': aweme_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_follow_list(sec_uid: str, type: str, maxcursor: int=0, count: int=50, mincursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user follower or following"
    count: max=50
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getfollow"
    querystring = {'sec_uid': sec_uid, 'type': type, }
    if maxcursor:
        querystring['maxCursor'] = maxcursor
    if count:
        querystring['count'] = count
    if mincursor:
        querystring['minCursor'] = mincursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_favorite_video_by_id(mixid: int, cursor: int=0, count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get favorite video by id
		**url doesn't seem to work please use get video details to get download link**"
    cursor: next page
        count: max=30
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getvideofavorite"
    querystring = {'mixId': mixid, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_user(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tiktok user info"
    
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getinfouser-v2"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_favorite_list(sec_uid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user favorite list id"
    
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getlistfavorite"
    querystring = {'sec_uid': sec_uid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video(keywords: str, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search video"
    cursor: max=5
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/searchvideo"
    querystring = {'keywords': keywords, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_list(aweme_id: int, cursor: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comment list"
    cursor: next page
        count: max=30
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/getcommentlist"
    querystring = {'aweme_id': aweme_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(keywords: str, count: int=5, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Search users by keyword**"
    count: max=5
        cursor: next page
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/searchusers"
    querystring = {'keywords': keywords, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_detail(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get links without watermarks quickly and easily**
		No support url image list"
    url: Support url: (ID: xxxxxxxxxxxx)
- https://www.tiktok.com/@username/video/xxxxxxxxxxxx
- https://vm.tiktok.com/xxxxxxxxxxxx/
- https://vt.tiktok.com/xxxxxxxxxxxx/
- xxxxxxxxxxxx
        
    """
    url = f"https://tiktok-video-no-watermark5.p.rapidapi.com/videodetail"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

