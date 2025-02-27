import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_tagged_posts(user_id: str, cursor: str=None, count: str='12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Tagged Posts"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        cursor: Cursor Token
        count: Number of Results
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/tagged-posts/"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_live_status(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Live Status"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/live-status/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_reels(user_id: str, count: int=12, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Reels"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        count: Number of Results
        max_id: Max ID
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/reels/"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(user_id: str, count: int=12, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Posts"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        count: Number of Results
        cursor: Cursor Token


        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/posts/"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_highlights(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Highlights"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/highlights/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_guides(user_id: str, count: int=12, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Guides"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        count: Number of Results
        max_id: Max ID
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/guides/"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following_hashtags(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Following Hashtags"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/following-hashtags/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Details By Username"
    username: Username
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/user/details-by-username/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guide_items(guide_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Guide Items"
    guide_id: Guide ID
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/guide/items/"
    querystring = {'guide_id': guide_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guide_details(guide_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Guide Details"
    guide_id: Guide ID
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/guide/details/"
    querystring = {'guide_id': guide_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, context: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    query: Search Query
        context: Search Type

Default: blended
        
    """
    url = f"https://instagram-v2.p.rapidapi.com/search/"
    querystring = {'query': query, }
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

