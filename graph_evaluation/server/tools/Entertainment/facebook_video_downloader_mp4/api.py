import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def facebook_video_downloader(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook video downloader API @ https://mp3downy.com/facebook-video-downloader-api"
    
    """
    url = f"https://facebook-video-downloader-mp4.p.rapidapi.com/facebook-video-downloader-api"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-video-downloader-mp4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

