import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def screenshot_your_page(url: str, type: str=None, blockads: bool=None, timeout: int=None, useragent: str=None, delay: int=None, height: int=None, disableanimations: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "( change .json()   to =>   .blob() response )"
    useragent: [Custom User Agent](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent), 'mobile', 'desktop'
        
    """
    url = f"https://web-capture1.p.rapidapi.com/capture"
    querystring = {'url': url, }
    if type:
        querystring['type'] = type
    if blockads:
        querystring['blockAds'] = blockads
    if timeout:
        querystring['timeout'] = timeout
    if useragent:
        querystring['userAgent'] = useragent
    if delay:
        querystring['delay'] = delay
    if height:
        querystring['height'] = height
    if disableanimations:
        querystring['disableAnimations'] = disableanimations
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-capture1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

