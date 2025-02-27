import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mediainfo_v2(short_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media info from short_code v2"
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/media_info_v2"
    querystring = {'short_code': short_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medias_v2(user_id: str, batch_size: str='30', max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's medias v2 version, batch_size range from 1 to 50."
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/medias_v2"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medias(user_id: str, batch_size: str='30', next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's medias, batch_size range from 1 to 50."
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/medias"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userinfo(user_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user info by user name."
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/user_info"
    querystring = {'user_name': user_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locationmedias(location_id: str, batch_size: int=20, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get medias from a specific location. Location id can be found at https://www.instagram.com/explore/locations/. For example https://www.instagram.com/explore/locations/303394066/chicago-union-station/ means location_id is 303394066."
    batch_size: No more than 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/location_medias"
    querystring = {'location_id': location_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtagmediasv2(hash_tag: str, batch_size: int=20, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get correct hashtag medias."
    batch_size: No more than 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/hash_tag_medias_v2"
    querystring = {'hash_tag': hash_tag, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchfollowing(user_id: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search someone's following."
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/search_following"
    querystring = {'user_id': user_id, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchfollowers(query: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search someone's followers."
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/search_followers"
    querystring = {'query': query, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stories(user_id: str='1154485247', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's stories."
    user_id: Range from 1 to 40.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/stories"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtagmedias(hash_tag: str, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag medias."
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/hash_tag_medias"
    querystring = {'hash_tag': hash_tag, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following(user_id: str, batch_size: int=12, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's following."
    batch_size: No more than 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/following"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers(user_id: str, batch_size: int=12, next_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get someone's followers."
    batch_size: No more than 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/followers"
    querystring = {'user_id': user_id, }
    if batch_size:
        querystring['batch_size'] = batch_size
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medialikers(short_code: str, next_cursor: str=None, batch_size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media's likers. batch_size range from 1 to 50."
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/media_likers"
    querystring = {'short_code': short_code, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    if batch_size:
        querystring['batch_size'] = batch_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mediacomments(short_code: str, next_cursor: str=None, batch_size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media's comments"
    batch_size: Range from 1 to 50.
        
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/media_comments"
    querystring = {'short_code': short_code, }
    if next_cursor:
        querystring['next_cursor'] = next_cursor
    if batch_size:
        querystring['batch_size'] = batch_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mediainfo(short_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media info from short_code"
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/media_info"
    querystring = {'short_code': short_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detailuserinfo(user_name: str='instagram', user_id: str='25025320', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail instagram user info."
    
    """
    url = f"https://instagram-scraper2.p.rapidapi.com/user_info_by_id"
    querystring = {}
    if user_name:
        querystring['user_name'] = user_name
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

