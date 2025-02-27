import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def username_to_id(username: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user id from username"
    
    """
    url = f"https://scraptik.p.rapidapi.com/username-to-id"
    querystring = {'username': username, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x_bogus_web_signature(user_agent: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate X-Bogus signature for Web."
    
    """
    url = f"https://scraptik.p.rapidapi.com/xbogus"
    querystring = {'user_agent': user_agent, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x_ladon_x_argus_generator(url: str, headers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sign TikTok api url, generate X-Ladon / X-Argus / X-Gorgon token and X-Khronos."
    
    """
    url = f"https://scraptik.p.rapidapi.com/xladon-xargus"
    querystring = {'url': url, 'headers': headers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x_gorgon_x_khronos_generator(url: str, headers: str, data: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sign TikTok api url, generate X-Gorgon token and X-Khronos."
    
    """
    url = f"https://scraptik.p.rapidapi.com/gorgon"
    querystring = {'url': url, 'headers': headers, }
    if data:
        querystring['data'] = data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_registeration(tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a new device."
    
    """
    url = f"https://scraptik.p.rapidapi.com/device-register"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiktok_url_shortener(url: str, device: str=None, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TikTok URL Shortener."
    
    """
    url = f"https://scraptik.p.rapidapi.com/shorten"
    querystring = {'url': url, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_without_watermark(tok_proxy: str=None, device: str=None, url: str='https://vm.tiktok.com/ZSJXgHCmR/', aweme_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video without watermark by url or aweme_id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/video-without-watermark"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if url:
        querystring['url'] = url
    if aweme_id:
        querystring['aweme_id'] = aweme_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music_posts(music_id: str, count: int=30, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music posts by music id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/web/music-posts"
    querystring = {'music_id': music_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user info by username."
    
    """
    url = f"https://scraptik.p.rapidapi.com/web/get-user"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts(secuid: str, cursor: str='0', count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user posts by sec user id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/web/user-posts"
    querystring = {'secUid': secuid, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover_music(device: str=None, tok_proxy: str=None, cookie: str=None, cursor: str='0', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover Music."
    
    """
    url = f"https://scraptik.p.rapidapi.com/discover-music"
    querystring = {}
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if cookie:
        querystring['cookie'] = cookie
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_posts(keyword: str, device: str=None, tok_proxy: str=None, offset: str='0', sort_type: int=0, count: str='20', cookie: str=None, publish_time: int=0, use_filters: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search posts."
    sort_type: - 0 = Relevance
- 1 = Most Liked
        publish_time: - 0 = All Time
- 1 = Yesterday
- 7 = This Week
- 30 = This Month
- 90 = Last 3 Months
- 180 = Last 6 Months
        use_filters: - 0 = No
- 1 = Yes
        
    """
    url = f"https://scraptik.p.rapidapi.com/search-posts"
    querystring = {'keyword': keyword, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if offset:
        querystring['offset'] = offset
    if sort_type:
        querystring['sort_type'] = sort_type
    if count:
        querystring['count'] = count
    if cookie:
        querystring['cookie'] = cookie
    if publish_time:
        querystring['publish_time'] = publish_time
    if use_filters:
        querystring['use_filters'] = use_filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(keyword: str, device: str=None, tok_proxy: str=None, count: str='20', cookie: str=None, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users."
    
    """
    url = f"https://scraptik.p.rapidapi.com/search-users"
    querystring = {'keyword': keyword, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if count:
        querystring['count'] = count
    if cookie:
        querystring['cookie'] = cookie
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_lives(keyword: str, device: str=None, tok_proxy: str=None, count: str='20', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search live streams"
    
    """
    url = f"https://scraptik.p.rapidapi.com/search-lives"
    querystring = {'keyword': keyword, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags(keyword: str, tok_proxy: str=None, device: str=None, count: str='20', cursor: str='0', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Hashtags."
    
    """
    url = f"https://scraptik.p.rapidapi.com/search-hashtags"
    querystring = {'keyword': keyword, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_sounds(keyword: str, tok_proxy: str=None, device: str=None, sort_type: int=0, use_filters: int=0, count: str='20', cursor: str='0', cookie: str=None, filter_by: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search sounds."
    sort_type: - 0 = Relevance
- 1 = Most used
- 2 = Most recent
- 3 = Shortest
- 4 = Longest
        use_filters: - 0 = No
- 1 = Yes
        filter_by: - 0 = All
- 1 = Title
- 2 = Creators
        
    """
    url = f"https://scraptik.p.rapidapi.com/search-sounds"
    querystring = {'keyword': keyword, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if sort_type:
        querystring['sort_type'] = sort_type
    if use_filters:
        querystring['use_filters'] = use_filters
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    if cookie:
        querystring['cookie'] = cookie
    if filter_by:
        querystring['filter_by'] = filter_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(device: str=None, tok_proxy: str=None, sec_user_id: str=None, cookie: str=None, max_cursor: str='0', user_id: str='6546356850533602319', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user likes by user id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/user-likes"
    querystring = {}
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if cookie:
        querystring['cookie'] = cookie
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comments_replies(aweme_id: str, comment_id: str, tok_proxy: str=None, device: str=None, cursor: str='0', count: int=10, cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get replies of a comment."
    
    """
    url = f"https://scraptik.p.rapidapi.com/list-comments-reply"
    querystring = {'aweme_id': aweme_id, 'comment_id': comment_id, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_search(keyword: str, tok_proxy: str=None, device: str=None, publish_time: int=0, sort_type: int=0, offset: str='0', count: str='20', use_filters: int=0, cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Search: videos, hashtags, sounds, users"
    publish_time: - 0 = All Time
- 1 = Yesterday
- 7 = This Week
- 30 = This Month
- 90 = Last 3 Months
- 180 = Last 6 Months
        sort_type: - 0 = Relevance
- 1 = Most Liked
        use_filters: - 0 = No
- 1 = Yes
        
    """
    url = f"https://scraptik.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if publish_time:
        querystring['publish_time'] = publish_time
    if sort_type:
        querystring['sort_type'] = sort_type
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if use_filters:
        querystring['use_filters'] = use_filters
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_posts(cid: str, tok_proxy: str=None, device: str=None, count: str='20', cookie: str=None, cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hashtag/Challenge Posts."
    
    """
    url = f"https://scraptik.p.rapidapi.com/hashtag-posts"
    querystring = {'cid': cid, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if count:
        querystring['count'] = count
    if cookie:
        querystring['cookie'] = cookie
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_posts(music_id: str, device: str=None, tok_proxy: str=None, cursor: int=0, count: int=18, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts by music_id"
    
    """
    url = f"https://scraptik.p.rapidapi.com/music-posts"
    querystring = {'music_id': music_id, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comments(aweme_id: str, device: str=None, tok_proxy: str=None, cookie: str=None, cursor: str='0', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list comments of a post."
    
    """
    url = f"https://scraptik.p.rapidapi.com/list-comments"
    querystring = {'aweme_id': aweme_id, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if cookie:
        querystring['cookie'] = cookie
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_followers(tok_proxy: str=None, device: str=None, sec_user_id: str=None, user_id: str='6802299750194643973', count: int=10, cookie: str=None, max_time: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of followers by user id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/list-followers"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if user_id:
        querystring['user_id'] = user_id
    if count:
        querystring['count'] = count
    if cookie:
        querystring['cookie'] = cookie
    if max_time:
        querystring['max_time'] = max_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(tok_proxy: str=None, device: str=None, username: str=None, user_id: str=None, sec_user_id: str='MS4wLjABAAAAxmiNgjKTNw0rn5WwweMsKe4WzgvbxRT_xeKhn-bA4O4HM3VS8piCMo88EXopqKh7', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user by id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/get-user"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post(aweme_id: str, tok_proxy: str=None, device: str=None, cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post by id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/get-post"
    querystring = {'aweme_id': aweme_id, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_you_feed(tok_proxy: str=None, device: str=None, region: str='US', max_cursor: str='0', min_cursor: str='0', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending posts for you."
    
    """
    url = f"https://scraptik.p.rapidapi.com/feed"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if region:
        querystring['region'] = region
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if min_cursor:
        querystring['min_cursor'] = min_cursor
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(device: str=None, tok_proxy: str=None, user_id: str='6546356850533602319', sec_user_id: str=None, count: int=10, max_cursor: str='0', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user posts by user id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/user-posts"
    querystring = {}
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if user_id:
        querystring['user_id'] = user_id
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if count:
        querystring['count'] = count
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music(music_id: str, device: str=None, tok_proxy: str=None, cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music by id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/get-music"
    querystring = {'music_id': music_id, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_stories(cookie: str, tok_proxy: str=None, device: str=None, user_id: str='6857828714567910406', username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user stories."
    
    """
    url = f"https://scraptik.p.rapidapi.com/user-stories"
    querystring = {'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following_feed(cookie: str, tok_proxy: str=None, device: str=None, region: str='US', max_cursor: str='0', min_cursor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get following feed."
    
    """
    url = f"https://scraptik.p.rapidapi.com/feed-following"
    querystring = {'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if region:
        querystring['region'] = region
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if min_cursor:
        querystring['min_cursor'] = min_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_following(device: str=None, tok_proxy: str=None, count: int=10, max_time: int=0, sec_user_id: str=None, user_id: str='6802299750194643973', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of following by user id."
    
    """
    url = f"https://scraptik.p.rapidapi.com/list-following"
    querystring = {}
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if count:
        querystring['count'] = count
    if max_time:
        querystring['max_time'] = max_time
    if sec_user_id:
        querystring['sec_user_id'] = sec_user_id
    if user_id:
        querystring['user_id'] = user_id
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_unique_id(unique_id: str, cookie: str, device: str=None, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check unique id if exist"
    
    """
    url = f"https://scraptik.p.rapidapi.com/unique-id-check"
    querystring = {'unique_id': unique_id, 'cookie': cookie, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def from_tiktok_notifications(min_time: str, cookie: str, max_time: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "From TikTok" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/from-tiktok"
    querystring = {'min_time': min_time, 'cookie': cookie, 'max_time': max_time, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_notifications(max_time: str, min_time: str, cookie: str, device: str=None, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "Followers" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/followers"
    querystring = {'max_time': max_time, 'min_time': min_time, 'cookie': cookie, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def likes_notifications(max_time: str, cookie: str, min_time: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "Likes" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/likes"
    querystring = {'max_time': max_time, 'cookie': cookie, 'min_time': min_time, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_notifications(max_time: str, min_time: str, cookie: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "Comments" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/comments"
    querystring = {'max_time': max_time, 'min_time': min_time, 'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mentions_tags_notifications(min_time: str, max_time: str, cookie: str, device: str=None, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "Mentions & Tags " account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/mentions-tags"
    querystring = {'min_time': min_time, 'max_time': max_time, 'cookie': cookie, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def q_a_notifications(min_time: str, cookie: str, max_time: str, device: str=None, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "Q&A" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/notifications/q-a"
    querystring = {'min_time': min_time, 'cookie': cookie, 'max_time': max_time, }
    if device:
        querystring['device'] = device
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_activity(cookie: str, max_time: str, min_time: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show "All Activity" account notifications."
    
    """
    url = f"https://scraptik.p.rapidapi.com/account-notice-lists"
    querystring = {'cookie': cookie, 'max_time': max_time, 'min_time': min_time, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login_using_sms_code(device: str, api_domain: str, code: str, mobile: str, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Login using SMS code."
    
    """
    url = f"https://scraptik.p.rapidapi.com/sms-login"
    querystring = {'device': device, 'api_domain': api_domain, 'code': code, 'mobile': mobile, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_sms(mobile: str, tok_proxy: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send SMS code to you mobile."
    
    """
    url = f"https://scraptik.p.rapidapi.com/send-sms"
    querystring = {'mobile': mobile, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def end_a_live_stream(stream_id: str, room_id: str, cookie: str, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "End a live stream."
    
    """
    url = f"https://scraptik.p.rapidapi.com/end-livestream"
    querystring = {'stream_id': stream_id, 'room_id': room_id, 'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def start_a_live_stream(stream_id: str, room_id: str, cookie: str, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Start a live stream."
    
    """
    url = f"https://scraptik.p.rapidapi.com/start-livestream"
    querystring = {'stream_id': stream_id, 'room_id': room_id, 'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live_stream_messages(room_id: str, tok_proxy: str=None, cookie: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live stream chat messages."
    
    """
    url = f"https://scraptik.p.rapidapi.com/livestream-messages"
    querystring = {'room_id': room_id, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if cookie:
        querystring['cookie'] = cookie
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_a_live_stream(cookie: str, title: str, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a new live stream."
    
    """
    url = f"https://scraptik.p.rapidapi.com/create-livestream"
    querystring = {'cookie': cookie, 'title': title, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def can_create_live_stream(cookie: str, tok_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if user can create live streams on TikTok."
    
    """
    url = f"https://scraptik.p.rapidapi.com/can-create-livestream"
    querystring = {'cookie': cookie, }
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_challenges(tok_proxy: str=None, device: str=None, cursor: str='0', count: int=10, region: str='US', cookie: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending challenges."
    
    """
    url = f"https://scraptik.p.rapidapi.com/category-list"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if region:
        querystring['region'] = region
    if cookie:
        querystring['cookie'] = cookie
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_creators(tok_proxy: str=None, device: str=None, cookie: str=None, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending creators."
    
    """
    url = f"https://scraptik.p.rapidapi.com/trending-creators"
    querystring = {}
    if tok_proxy:
        querystring['tok-proxy'] = tok_proxy
    if device:
        querystring['device'] = device
    if cookie:
        querystring['cookie'] = cookie
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

