import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_youtube_subtitles(v: str, format: str='xml', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a link to download subtitle of video"
    format: It can be xml, ttml, vtt, srv1, srv2, srv3
        
    """
    url = f"https://metube-one-get-link-download-videos-and-subtitles.p.rapidapi.com/captions"
    querystring = {'v': v, }
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metube-one-get-link-download-videos-and-subtitles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_info(v: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video info and link downloadable."
    
    """
    url = f"https://metube-one-get-link-download-videos-and-subtitles.p.rapidapi.com/info"
    querystring = {'v': v, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metube-one-get-link-download-videos-and-subtitles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

