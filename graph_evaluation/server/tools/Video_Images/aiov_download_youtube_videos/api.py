import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_video_streams_youtube_instagram_facebook_snapchat_tiktok_koo_and_linkedin(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get youtube, Instagram, Facebook, Snapchat, TikTok, Koo, and LinkedIn videos Data like title, thumb, URL, size, type, etc."
    
    """
    url = f"https://aiov-download-youtube-videos.p.rapidapi.com/GetVideoDetails"
    querystring = {'URL': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiov-download-youtube-videos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

