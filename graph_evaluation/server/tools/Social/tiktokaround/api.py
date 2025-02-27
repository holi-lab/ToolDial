import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_videos_user_tiktok_id_id(is_id: str, is_newest: str='true', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos of user by tiktok id
		Params: 
		- id (string) : tiktok id of user
		- is_newest (bool) : Is get newest videos ( default: false)
		- offset (int)"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/videos/user/tiktok-id/{is_id}"
    querystring = {}
    if is_newest:
        querystring['is_newest'] = is_newest
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_tiktok_id_tiktok_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user by tiktok id
		Params: 
		- tiktok_id (string) : tiktok id of user"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/users/tiktok-id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_videos_tiktok_id_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video by tiktok id
		Params:
		- id (string) : tiktok id of video"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/videos/tiktok-id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music(title: str='꽃', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music by title
		Params:
		- title (string): name of the music"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/music"
    querystring = {}
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_username_username(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user by username
		Params:
		- username (string): username of user"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/users/username/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_videos_user_username_id(is_id: str, is_newest: str='true', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos of user by username
		Params: 
		- id (string) : tiktok id of user
		- is_newest (bool) : Is get newest videos ( default: false)
		- offset (int)"
    
    """
    url = f"https://tiktokaround5.p.rapidapi.com/v1/api/videos/user/username/{is_id}"
    querystring = {}
    if is_newest:
        querystring['is_newest'] = is_newest
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktokaround5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

