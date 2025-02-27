import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_video_downlad(videourl: str='https://www.youtube.com/watch?v=dQw4w9WgXcQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "youtube video downlad from youtube"
    
    """
    url = f"https://youtube-video-download.p.rapidapi.com/video"
    querystring = {}
    if videourl:
        querystring['videourl'] = videourl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

