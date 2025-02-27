import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def full_info_with_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tiktok videos with full info 
		video without watermark
		cover / music
		share / likes / comments
		author  info  like connected  youtube channel / IG  / avatar thumbnail / statistics about the video including comments number / likes 
		support for  all tiktok links ( vm.tiktok.com / vt.tiktok.com / m.tiktok.com / web / t.tiktok.com)"
    
    """
    url = f"https://tiktok-full-info-without-watermark.p.rapidapi.com/info/index"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-full-info-without-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_without_watermark_cover_music(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "original video without watermark/cover/ music / other info about the user and the video"
    
    """
    url = f"https://tiktok-full-info-without-watermark.p.rapidapi.com/vid/index"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-full-info-without-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

