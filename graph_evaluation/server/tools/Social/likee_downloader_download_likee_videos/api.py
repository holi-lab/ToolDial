import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_videos(uid: str, pagesize: str='10', count: str='4', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all videos about a specific user on Likee."
    uid: You will find in Get User Info endpoint.
        
    """
    url = f"https://likee-downloader-download-likee-videos.p.rapidapi.com/user"
    querystring = {'uid': uid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "likee-downloader-download-likee-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all information about a specific video on Likee."
    
    """
    url = f"https://likee-downloader-download-likee-videos.p.rapidapi.com/postid"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "likee-downloader-download-likee-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all information about a specific user on Likee."
    username: Example: MariamHany
Extract username from url: https://likee.video/@MariamHany
        
    """
    url = f"https://likee-downloader-download-likee-videos.p.rapidapi.com/likeeId"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "likee-downloader-download-likee-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_without_watermark(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all information about a specific video on Likee with watermark/without watermark"
    
    """
    url = f"https://likee-downloader-download-likee-videos.p.rapidapi.com/process"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "likee-downloader-download-likee-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

