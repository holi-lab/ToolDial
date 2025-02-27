import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_a_youtube_video_in_any_format(video_id: str, format: str, stime: str='00:01:00', etime: str='00:02:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a YouTube video in any format !"
    
    """
    url = f"https://download-video-youtube1.p.rapidapi.com/{format}/{video_id}"
    querystring = {}
    if stime:
        querystring['stime'] = stime
    if etime:
        querystring['etime'] = etime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "download-video-youtube1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

