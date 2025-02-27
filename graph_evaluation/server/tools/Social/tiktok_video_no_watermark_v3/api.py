import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tiktok_video_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tiktok video full info. HD Quality, No Watermark. Fast.
		Support Tiktok & Douyin.
		Support Getting Image List.
		Support Tiktok Stories."
    url: use tiktok id: 7213756991653547269 
or https://vm.tiktok.com/ZTRvK8Fn4/
or https://www.tiktok.com/@tiktok/video/7213756991653547269
        
    """
    url = f"https://tiktok-video-no-watermark6.p.rapidapi.com/tik"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-video-no-watermark6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

