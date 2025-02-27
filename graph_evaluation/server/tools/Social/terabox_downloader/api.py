import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def info_link_data(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Data Terabox Share Link"
    url: Example: https://terabox.com/s/1gw8aGb75AWIp0fkfd3VaAgs
        
    """
    url = f"https://terabox-downloader.p.rapidapi.com/info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "terabox-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stream_m3u8_format(q: int, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stream by url with m3u8 format"
    q: Quality, write 480 for 480p, 720 for 720p and 1080 for 1080p
        url: Example: https://terabox.com/s/1ZyRIwRGKKvHq5HWQtsV1oAs
        
    """
    url = f"https://terabox-downloader.p.rapidapi.com/stream"
    querystring = {'q': q, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "terabox-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_download_link(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Download Link"
    url: Example: https://terabox.com/s/1gw8aGb75AWIp0fkfd3VaAgs
        
    """
    url = f"https://terabox-downloader.p.rapidapi.com/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "terabox-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

