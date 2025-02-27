import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_video_info(videoid: str, cache_control: str='no-cache', connection: str='keep-alive', accept: str='text/html,application/xhtml+xml,application/xml;q=0.9', pragma: str='no-cache', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dzdadaz"
    
    """
    url = f"https://youtube-video-info2.p.rapidapi.com/votes"
    querystring = {'videoId': videoid, }
    if cache_control:
        querystring['Cache-Control'] = cache_control
    if connection:
        querystring['Connection'] = connection
    if accept:
        querystring['Accept'] = accept
    if pragma:
        querystring['Pragma'] = pragma
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-info2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_info(videoid: str, cache_control: str='no-cache', connection: str='keep-alive', accept: str='text/html,application/xhtml+xml,application/xml;q=0.9', pragma: str='no-cache', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dzdadaz"
    
    """
    url = f"https://youtube-video-info2.p.rapidapi.com/votes"
    querystring = {'videoId': videoid, }
    if cache_control:
        querystring['Cache-Control'] = cache_control
    if connection:
        querystring['Connection'] = connection
    if accept:
        querystring['Accept'] = accept
    if pragma:
        querystring['Pragma'] = pragma
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-info2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

