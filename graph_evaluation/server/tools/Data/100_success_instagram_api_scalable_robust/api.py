import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def post_likes(mediaid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post likes list"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/media/{mediaid}/likers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_info(media_id: str, mediaid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media info"
    media_id: needs to be the same as mediaId in url
        
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/media/{mediaid}/info"
    querystring = {'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed(tagname: str, rank_token: str, max_id: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag feed"
    rank_token: This parameter is a UUID version 5, and for each request that use it you should pass a newly generated UUID. If the request is of pagination type, use the same `rank_token` for all pagination requests.

For example, when you request the `paris` hashtag feed and the generated UUID is `f0d8368d-85e2-54fb-73c4-2d60374295e3`, all the pagination requests of `paris` hashtag will use the same UUID:

```
GET /instagram/v1/feed/tag/paris?rank_token=f0d8368d-85e2-54fb-73c4-2d60374295e3
GET /instagram/v1/feed/tag/paris?rank_token=f0d8368d-85e2-54fb-73c4-2d60374295e3&max_id=QVFCQ0ZDMUlNOEw0X3dyNGJQSXR0UVVNaU9yM2tNdzlFN2ZyMnJDY1V2VXhTemEzbTQzTmxfRTNhY3pHUlFMT0kwQ2RVMTNqSy1DRXJadm1SWW41THhQTw==
GET /instagram/v1/feed/tag/paris?rank_token=f0d8368d-85e2-54fb-73c4-2d60374295e3&max_id=...
```

Then, when you request for a different hashtag, let's say `israel`, you'll have to generate a new UUID:

```
GET /instagram/v1/feed/tag/israel?rank_token=35adbb92-df2c-4ec3-909e-ddaa081d2b39
GET /instagram/v1/feed/tag/israel?rank_token=35adbb92-df2c-4ec3-909e-ddaa081d2b39&max_id=...
```


        
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/feed/tag/{tagname}"
    querystring = {'rank_token': rank_token, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def child_comments(commentid: int, mediaid: int, max_id: str=None, min_id: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get child comments list"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/media/{mediaid}/comments/{commentid}/child_comments/"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_info(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results of hashtag search by keyword"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/tags/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user search results of keyword"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/users/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(userid: int, order: str='default', query: str=None, enable_groups: str='true', max_id: str=None, search_surface: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers list"
    query: Filter followers by username
        max_id: Use \\\\\\\"next_max_id\\\\\\\" as the value for pagination
        
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/friendships/{userid}/followers"
    querystring = {}
    if order:
        querystring['order'] = order
    if query:
        querystring['query'] = query
    if enable_groups:
        querystring['enable_groups'] = enable_groups
    if max_id:
        querystring['max_id'] = max_id
    if search_surface:
        querystring['search_surface'] = search_surface
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed(userid: int, max_id: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user feed"
    max_id: Use \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"next_max_id\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" as the value for pagination
        
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/feed/user/{userid}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info(userid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data of user profile"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/users/{userid}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_comments(mediaid: int, min_id: str='{}', max_id: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post comments list"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/media/{mediaid}/comments"
    querystring = {}
    if min_id:
        querystring['min_id'] = min_id
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fbsearch_places(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results of fbsearch-places endpoint"
    
    """
    url = f"https://100-success-instagram-api-scalable-robust.p.rapidapi.com/instagram/v1/fbsearch/places"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "100-success-instagram-api-scalable-robust.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

