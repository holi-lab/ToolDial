import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_pinterest_video_image_link(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Pinterest Video/Image Link"
    url: Both Full and Short Pin support. 
        
    """
    url = f"https://pinterest-video-and-image-downloader.p.rapidapi.com/pinterest"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinterest-video-and-image-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pins_by_pinterest_user(username: str, bookmark: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Image and Video of Pinterest User"
    username: Example: viratkohli
        bookmark: A bookmark is used to access the next page of data. Each page typically returns 20-25 pins, and you receive the bookmark value in response to the first page's data.
        
    """
    url = f"https://pinterest-video-and-image-downloader.p.rapidapi.com/pinterest-user"
    querystring = {'username': username, }
    if bookmark:
        querystring['bookmark'] = bookmark
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinterest-video-and-image-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

