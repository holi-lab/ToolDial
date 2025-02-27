import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_comment_find_get(by_facebook_id: str=None, by_object_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Comment by UID or by_object_id(PageID, GroupID) "
    
    """
    url = f"https://facebook-user-info.p.rapidapi.com/comment/find"
    querystring = {}
    if by_facebook_id:
        querystring['by_facebook_id'] = by_facebook_id
    if by_object_id:
        querystring['by_object_id'] = by_object_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-user-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_feed_find_get(by_facebook_id: str=None, by_object_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Feed by UID or by_object_id(PageID, GroupID) "
    
    """
    url = f"https://facebook-user-info.p.rapidapi.com/feed/find"
    querystring = {}
    if by_facebook_id:
        querystring['by_facebook_id'] = by_facebook_id
    if by_object_id:
        querystring['by_object_id'] = by_object_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-user-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_user_find_get(by_facebook_id: str=None, by_phone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://facebook-user-info.p.rapidapi.com/user/find"
    querystring = {}
    if by_facebook_id:
        querystring['by_facebook_id'] = by_facebook_id
    if by_phone:
        querystring['by_phone'] = by_phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-user-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

