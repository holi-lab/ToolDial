import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_profile_with_base64_image_beta(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Image with base64 image. This is beta version. Please report to us for any issues."
    
    """
    url = f"https://easy-instagram-service.p.rapidapi.com/username-with-base64-image"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-instagram-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_instagram_profile_by_username(username: str, random: str='x8n3nsj2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram follower,following, average like, last post."
    username: Instagram Username
        
    """
    url = f"https://easy-instagram-service.p.rapidapi.com/username"
    querystring = {'username': username, }
    if random:
        querystring['random'] = random
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-instagram-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

