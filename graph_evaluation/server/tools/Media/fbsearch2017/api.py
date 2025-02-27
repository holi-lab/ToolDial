import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def facebook_search_photos_in(query: str, timeout: int, limit: int, parallel: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook: get profiles using Graph Search"
    query: Graph Query
        timeout: Response Timeout
        limit: Limit
        parallel: Parallel Search (0 or 1)
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/facebook/search_photos_in"
    querystring = {'query': query, 'timeout': timeout, 'limit': limit, }
    if parallel:
        querystring['parallel'] = parallel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_user(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FB User Info by ID"
    id: FB ID
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/facebook/user"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_friends_v2(is_id: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User Friends List by user FB ID"
    id: FB User ID
        limit: Response limit (max 2000)
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/facebook/friends/v2"
    querystring = {'id': is_id, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fbparser_about_video(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FB Video Info (and 5 neighbor) by Video ID"
    id: FB Video ID
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/fbparser/about_video"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fbparser_photo(is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FB Photo Info (and 5 neighbor) by Photo ID"
    is_id: FB Photo ID
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/fbparser/photo"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_likes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FB public's posts likes list by post's ID"
    id: FB public post ID
        
    """
    url = f"https://fbsearch-fbsearch2017-v1.p.rapidapi.com/facebook/likes"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fbsearch-fbsearch2017-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

