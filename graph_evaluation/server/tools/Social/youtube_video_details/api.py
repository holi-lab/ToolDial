import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_video_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube Video Details"
    id: **Video ID** or **Video URL**

e.g. `SmM0653YvXU`
e.g. `https://youtu.be/SmM0653YvXU`
e.g. `https://www.youtube.com/watch?v=SmM0653YvXU`
        
    """
    url = f"https://youtube-video-details.p.rapidapi.com/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

