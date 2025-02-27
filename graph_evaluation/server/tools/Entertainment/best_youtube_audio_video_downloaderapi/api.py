import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_json_api_copy(url: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch YouTube JSON data to download or access its video and audio files. This is the best and fastest API with low fees."
    
    """
    url = f"https://best-youtube-audio-video-downloaderapi.p.rapidapi.com/"
    querystring = {'url': url, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-youtube-audio-video-downloaderapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_json_api(url: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch YouTube JSON data to download or access its video and audio files. This is the best and fastest API with low fees."
    
    """
    url = f"https://best-youtube-audio-video-downloaderapi.p.rapidapi.com/"
    querystring = {'url': url, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-youtube-audio-video-downloaderapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

