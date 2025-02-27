import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_info_from_youtube_video(video_url: str='www.youtube.com/watch?v=K09xQuFdDuA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send request with url of youtube video and get information about channel, comments below video and video description."
    
    """
    url = f"https://youtube-video-information.p.rapidapi.com/comments"
    querystring = {}
    if video_url:
        querystring['video_url'] = video_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

