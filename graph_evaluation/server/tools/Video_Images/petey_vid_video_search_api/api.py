import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_search_api(q: str, dur: str=None, sort: str=None, media: str='Y', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to search Petey Vid index/repository"
    
    """
    url = f"https://petey-vid-video-search-api.p.rapidapi.com/"
    querystring = {'q': q, }
    if dur:
        querystring['dur'] = dur
    if sort:
        querystring['sort'] = sort
    if media:
        querystring['media'] = media
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "petey-vid-video-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

