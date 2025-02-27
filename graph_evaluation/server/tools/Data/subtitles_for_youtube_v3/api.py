import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_subtitle_in_text_format(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get subtitle in TEXT format"
    
    """
    url = f"https://subtitles-for-youtube1.p.rapidapi.com/GetTextsubtitles"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subtitle_in_srt_format(video_id: str='Wrald_EZgDQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get subtitle in SRT format"
    
    """
    url = f"https://subtitles-for-youtube1.p.rapidapi.com/GetWebVTTsubtitles"
    querystring = {}
    if video_id:
        querystring['video_id'] = video_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subtitle_in_json_format(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Subtitle in JSON format"
    
    """
    url = f"https://subtitles-for-youtube1.p.rapidapi.com/GetJSONsubtitles"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

