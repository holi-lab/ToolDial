import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def without_watermark(url: str, hd: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get without watermark video for url."
    hd: get HD Video (1080p)

`response.data.hdplay`
        
    """
    url = f"https://tiktok-download-without-watermark.p.rapidapi.com/analysis"
    querystring = {'url': url, }
    if hd:
        querystring['hd'] = hd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-download-without-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

