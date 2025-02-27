import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_users_by_query(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a list of Users by Query (string occurrence)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/users_search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags_by_query(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a list of Hashtags by Query (string occurrence)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/hashtag_search"
    querystring = {'hashtag': hashtag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_s_feed_by_its_name(hashtag: str, next_max_id: str='QVFEdFctaHdyal91U0lfeU1TOGJiOU5KMHBIeXpSbmVwc1UtTGZYckg0US1Gc3dWY201VjFfY0tIMjJMd0oxaDUzV3hPeVFaeWw0OFVIOGFMc3Zva2JqNw==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET hashtag's  Feed by its name."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/"
    querystring = {'hashtag': hashtag, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_post_story_igtv_reel_by_media_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET any Media (Post, Story (from highlight also), IGTV, or Reel) by its media_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/media"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def highlight_s_media_by_highlight_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Highlight's Media by its highlight_id 
		e.g. you can get it from a URL: https://instagram.com/stories/highlights/17920266223564403/ where 17920266223564403 is id"
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/highlight"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_igtv_reel_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Post, IGTV, or Reel by its shortcode."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/post"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_for_media_by_media_id(is_id: int, next_min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Comments for Media (Post / IGTV) by its media_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/comments"
    querystring = {'id': is_id, }
    if next_min_id:
        querystring['next_min_id'] = next_min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_igtvs_by_user_id(user_id: int, next_max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's IGTVs by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/igtv"
    querystring = {'user_id': user_id, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_highlights_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's Highlights by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/highlights"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_stories_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's Stories by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/stories"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_feed_by_user_id(user_id: int, next_max_id: str='2530701951199222173_13460080', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's Feed by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/feed"
    querystring = {'user_id': user_id, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_followings_by_user_id(user_id: int, next_max_id: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's Followings by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/following"
    querystring = {'user_id': user_id, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_followers_by_user_id(user_id: int, next_max_id: str='QVFDdER2LUxBd1lGcjFhbVVpMVR6QnNpdDg5c1V5MGJZSnQ0MFdXOEtfcnVLQ0J2WFhYU2dSbzdYX1FENWNzQzdoRlVNZnBUbEFxcDJrbl9UNEhlVHJSOQ==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's Followers by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/followers"
    querystring = {'user_id': user_id, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_profile_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Profile metadata by their user_id (pk)."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/profile"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_profile_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Profile metadata by their username."
    
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/profile"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_username_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Username by their user_id (pk)."
    user_id: Instagram ID
        
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/username_by_id"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_user_id_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's User_id (pk) by their username."
    username: User profile's username
        
    """
    url = f"https://instagram-api-cheap-best-performance.p.rapidapi.com/user_id_by_username"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-cheap-best-performance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

