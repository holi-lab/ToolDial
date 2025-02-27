import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_info(url: str, debug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info video tiktok, douyin"
    
    """
    url = f"https://tiktok-download-video-no-watermark2.p.rapidapi.com/api/v1/tiktok/video/info"
    querystring = {'url': url, 'debug': debug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video-no-watermark2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

