import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tiktok_douyin_video_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TikTok/Douyin public videos full info (video no watermark included)"
    url: TikTok video url to download.

Example inputs:
https://www.tiktok.com/@godownloader/video/6985792919672229146
https://vt.tiktok.com/ZGJBQHoHA/
https://m.tiktok.com/v/6985792919672229146.html
https://v.douyin.com/FcDBQD9/
        
    """
    url = f"https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-video-no-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

