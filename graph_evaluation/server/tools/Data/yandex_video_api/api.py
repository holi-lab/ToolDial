import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_video_from_yandex_search_results(page: int, query: str, domain: str, videoduration: str=None, region: str=None, videorecent: bool=None, videohd: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use advanced filtering to get specific video from yandex! Remember to set from which domain you want data. Default is yandex.com but  yandex.ru, .by, .kz, .uz are also available. Set page number in order to get more results. 0 is the first page, 1 is second and so on."
    page: 0 - first page
1 - second page
...
        domain: get search results from:
- yandex.com
- yandex.ru
- yandex.by
- yandex.kz
- yandex.uz
        region: e.g. Paris, France
        
    """
    url = f"https://yandex-video-api.p.rapidapi.com/getvideo"
    querystring = {'page': page, 'query': query, 'domain': domain, }
    if videoduration:
        querystring['videoDuration'] = videoduration
    if region:
        querystring['region'] = region
    if videorecent:
        querystring['videoRecent'] = videorecent
    if videohd:
        querystring['videoHd'] = videohd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-video-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets server time."
    
    """
    url = f"https://yandex-video-api.p.rapidapi.com/getservertime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-video-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

