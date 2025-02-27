import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_user_followers(user_id: str, sec_uid: str, proxy: str=None, offset: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followers List"
    user_id: Required if sec_uid is not present
        sec_uid: Required if user_id is not present
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user/follower/list"
    querystring = {'user_id': user_id, 'sec_uid': sec_uid, }
    if proxy:
        querystring['proxy'] = proxy
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_musics(keyword: str, proxy: str=None, count: int=10, offset: int=0, search_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Musics By Keyword"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/search/music"
    querystring = {'keyword': keyword, }
    if proxy:
        querystring['proxy'] = proxy
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if search_id:
        querystring['search_id'] = search_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags(keyword: str, proxy: str=None, count: str='10', offset: int=0, search_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Hashtags"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/search/challenge"
    querystring = {'keyword': keyword, }
    if proxy:
        querystring['proxy'] = proxy
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if search_id:
        querystring['search_id'] = search_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_search(keyword: str, search_id: str=None, count: int=10, offset: str=None, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Search: videos, hashtags, sounds, users"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/search/general"
    querystring = {'keyword': keyword, }
    if search_id:
        querystring['search_id'] = search_id
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(keyword: str, count: int=10, offset: int=None, proxy: str=None, search_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Users By Keyword"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/search/user"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if proxy:
        querystring['proxy'] = proxy
    if search_id:
        querystring['search_id'] = search_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_challenge(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Challenge"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/challenge/{hashtag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_videos(keyword: str, offset: int=0, count: int=10, proxy: str=None, search_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Videos By Keyword"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/search/video"
    querystring = {'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if proxy:
        querystring['proxy'] = proxy
    if search_id:
        querystring['search_id'] = search_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/music/{music_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_followings(user_id: str, sec_uid: str, proxy: str=None, offset: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followings List"
    user_id: Required if sec_uid is not present
        sec_uid: Required if user_id is not present
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user/following/list"
    querystring = {'user_id': user_id, 'sec_uid': sec_uid, }
    if proxy:
        querystring['proxy'] = proxy
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_posts(music_id: str, count: int=10, offset: int=0, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Posts By Music"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/music/{music_id}/posts"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_posts(hashtag: str, count: int=10, cursor: int=0, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Posts By Challenge Name"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/challenge/{hashtag}/posts"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_awemes_by_challenge_id(challenge_id: str, proxy: str=None, cursor: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Awemes By Challenge ID"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/challenge/{challenge_id}/aweme"
    querystring = {}
    if proxy:
        querystring['proxy'] = proxy
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comments(aweme_id: str, count: int=10, cursor: int=0, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Comments By Post"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/comments"
    querystring = {'aweme_id': aweme_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post(aweme_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Post Detail"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/post/{aweme_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_creators(proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending By Creators"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/creator/recommend"
    querystring = {}
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_replies_by_comment(aweme_id: str, comment_id: str, count: int=10, cursor: int=0, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Replies By Comment"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/replies"
    querystring = {'aweme_id': aweme_id, 'comment_id': comment_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_challenges(cursor: int=0, proxy: str=None, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Challenges"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/category/list"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if proxy:
        querystring['proxy'] = proxy
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(sec_uid: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User"
    sec_uid: Required if user_id is not present
        user_id: Required if sec_uid is not present
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user"
    querystring = {'sec_uid': sec_uid, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(sec_uid: str, user_id: str, cursor: str='0', count: int=10, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Posts By User"
    user_id: Required if sec_uid is not present
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user/posts"
    querystring = {'sec_uid': sec_uid, 'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_posts(user_id: str, playlist_id: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts by User and Playlist"
    user_id: User Id or User Secret Uid
        playlist_id: Playlist Id
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user/{user_id}/playlist/{playlist_id}/posts"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def username_to_user_id(username: str, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User By Username"
    
    """
    url = f"https://tiktok-private1.p.rapidapi.com/user/get-by-username"
    querystring = {'username': username, }
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_without_watermark(url: str, aweme_id: str, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Video Without Watermark"
    url: Required if aweme_id is not present
        aweme_id: Required if url is not present
        
    """
    url = f"https://tiktok-private1.p.rapidapi.com/video/without-watermark"
    querystring = {'url': url, 'aweme_id': aweme_id, }
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-private1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

