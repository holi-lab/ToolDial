import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_user(count: str, keywords: str, cursor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get user list by keywords"
    
    """
    url = f"https://tiktok-video-no-watermark8.p.rapidapi.com/searchUser"
    querystring = {'count': count, 'keywords': keywords, 'cursor': cursor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tiktok_video_info(url: str, hd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tiktok video full info. HD Quality, No Watermark. Fast.
		Support Tiktok & Douyin.
		Support Getting Image List.
		Support Tiktok Stories."
    
    """
    url = f"https://tiktok-video-no-watermark8.p.rapidapi.com/getVideo"
    querystring = {'url': url, 'hd': hd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

