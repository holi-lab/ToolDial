import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def guide_items(guide_id: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "guide-items"
    guide_id: Guide ID
        max_id: Max ID
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/guide/items/"
    querystring = {'guide_id': guide_id, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guide_details(guide_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "guide-details"
    guide_id: Guide ID
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/guide/details/"
    querystring = {'guide_id': guide_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tagged_posts(user_id: str, count: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-tagged-posts"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        count: Number of Results
        cursor: Cursor Token
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/tagged-posts/"
    querystring = {'user_id': user_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_reels(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-reels"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/reels/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-posts"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/posts/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_live_status(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-live-status"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/live-status/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_highlights(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-highlights"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/highlights/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_guides(user_id: str, max_id: str=None, count: str='12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-guides"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        max_id: Max ID
        count: Number of Results
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/guides/"
    querystring = {'user_id': user_id, }
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following_hashtags(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-following-hashtags"
    user_id: User ID

Use the User Details By Username endpoint to find the user ID from a username.
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/following-hashtags/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-details-by-username"
    username: username
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/user/details-by-username/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, context: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    query: Search Query
        context: Search Type

Default: blended
        
    """
    url = f"https://instagram-data12.p.rapidapi.com/search/"
    querystring = {'query': query, }
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

