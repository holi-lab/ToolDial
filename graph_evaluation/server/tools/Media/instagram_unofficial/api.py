import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def login(username: str, password: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Login using username and password."
    
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/login"
    querystring = {'username': username, 'password': password, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_user(session_key: str, insta_proxy: str=None, username: str='instagram', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user by user_id."
    
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/get-user"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_igtv(query: str, session_key: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search IGTV."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/search-igtv"
    querystring = {'query': query, 'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_tags(query: str, session_key: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search tags."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/search-tags"
    querystring = {'query': query, 'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(query: str, session_key: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/search-users"
    querystring = {'query': query, 'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tag_posts(session_key: str, tag: str, insta_proxy: str=None, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts by tag."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/tag-posts"
    querystring = {'session_key': session_key, 'tag': tag, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_search(query: str, session_key: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Search (hashtags, places, users)."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/search"
    querystring = {'query': query, 'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(session_key: str, insta_proxy: str=None, username: str=None, max_id: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/user-followers"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if max_id:
        querystring['max_id'] = max_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(session_key: str, insta_proxy: str=None, user_id: str=None, username: str=None, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user following."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/user-following"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(session_key: str, insta_proxy: str=None, username: str=None, max_id: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user posts."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/user-posts"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if max_id:
        querystring['max_id'] = max_id
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_stories(session_key: str, insta_proxy: str=None, username: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user stories."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/user-stories"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_tag(tag: str, insta_proxy: str=None, session_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a tag."
    
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/get-tag"
    querystring = {'tag': tag, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if session_key:
        querystring['session_key'] = session_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unlike_a_post(media_id: str, session_key: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Unlike a post by id."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/unlike-post"
    querystring = {'media_id': media_id, 'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def follow(session_key: str, insta_proxy: str=None, username: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Follow a user by username or user_id."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/follow"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def like_a_post(session_key: str, media_id: str, insta_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Like a post by id."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/like-post"
    querystring = {'session_key': session_key, 'media_id': media_id, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unfollow(session_key: str, insta_proxy: str=None, username: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Unfollow a user by username or user_id."
    session_key: Use our login api to get the session_key
        
    """
    url = f"https://instagram-unofficial.p.rapidapi.com/unfollow"
    querystring = {'session_key': session_key, }
    if insta_proxy:
        querystring['insta-proxy'] = insta_proxy
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

