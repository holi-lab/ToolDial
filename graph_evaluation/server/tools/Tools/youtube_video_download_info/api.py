import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_video_info(is_id: str, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the video download links and other relevant information in the JSON format."
    id: Youtube Video Id.
        geo: Country code in ISO 3166 format of the end user.
        
    """
    url = f"https://youtube-video-download-info.p.rapidapi.com/dl"
    querystring = {'id': is_id, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-download-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

