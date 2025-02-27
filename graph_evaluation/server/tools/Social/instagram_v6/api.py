import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mediainfo_v2(short_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get media info v2"
    
    """
    url = f"https://instagram28.p.rapidapi.com/media_info_v2"
    querystring = {'short_code': short_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userinfo(user_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user info by user name, response with old format. result -> user."
    
    """
    url = f"https://instagram28.p.rapidapi.com/user_info"
    querystring = {'user_name': user_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getusernamebyuserid(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get username by user id"
    
    """
    url = f"https://instagram28.p.rapidapi.com/username"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtagmedias(hash_tag: str, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag medias"
    
    """
    url = f"https://instagram28.p.rapidapi.com/hash_tag_medias"
    querystring = {'hash_tag': hash_tag, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medias(user_id: str, batch_size: int=20, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's medias, batch_size range from 1 to 50"
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram28.p.rapidapi.com/medias"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mediainfo(short_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get media info"
    
    """
    url = f"https://instagram28.p.rapidapi.com/media_info"
    querystring = {'short_code': short_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medialikers(short_code: str, next_cursor: str=None, batch_size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one media's likers, batch_size range from 1 to 50."
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram28.p.rapidapi.com/media_likers"
    querystring = {'short_code': short_code, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    if batch_size:
        querystring['batch_size'] = batch_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserstories(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get stories by user id"
    
    """
    url = f"https://instagram28.p.rapidapi.com/stories"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following(user_id: str, next_cursor: str=None, batch_size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get someone's following"
    batch_size: Custom value can only be set by paying user.
        
    """
    url = f"https://instagram28.p.rapidapi.com/following"
    querystring = {'user_id': user_id, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    if batch_size:
        querystring['batch_size'] = batch_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers(user_id: str, batch_size: int=20, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get someone's followers"
    batch_size: Custom value can only be set by paying user.
        
    """
    url = f"https://instagram28.p.rapidapi.com/followers"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchfollowers(user_id: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search someone's followers"
    
    """
    url = f"https://instagram28.p.rapidapi.com/search_followers"
    querystring = {'user_id': user_id, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchfollowing(user_id: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search someone's following"
    
    """
    url = f"https://instagram28.p.rapidapi.com/search_following"
    querystring = {'user_id': user_id, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsuggesteduser(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get suggested user by user id"
    
    """
    url = f"https://instagram28.p.rapidapi.com/suggested_user"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mediacomments(short_code: str, batch_size: str='20', next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one media's comments, batch_size range from 1 to 50."
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram28.p.rapidapi.com/media_comments"
    querystring = {'short_code': short_code, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getusernamebyuserid_v2(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get username by user id"
    
    """
    url = f"https://instagram28.p.rapidapi.com/username_v2"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

