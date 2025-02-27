import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_screenshot(url: str, accept: str='image/png', height: int=-1, mobile: bool=None, allocated_time: int=2, width: int=1024, base64: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a screenshot from a webpage url."
    url: URL of the webpage
        height: Height of the screenshot in pixels. If you want to take a full screenshot, set this parameter to -1. Maximum value: 10000.
        mobile: Should we emulate a mobile device?
        allocated_time: Time allocated to load the webpage, in seconds. Maximum value: 10.0.
        width: Width of the screenshot in pixels. Maximum value: 2048.
        base64: If 0, the API returns a PNG image. If 1, the API returns the image encoded in base 64 format (in that case, it might be a good idea to change the accept header to \"text/html\").
        
    """
    url = f"https://url-to-screenshot.p.rapidapi.com/get"
    querystring = {'url': url, }
    if accept:
        querystring['Accept'] = accept
    if height:
        querystring['height'] = height
    if mobile:
        querystring['mobile'] = mobile
    if allocated_time:
        querystring['allocated_time'] = allocated_time
    if width:
        querystring['width'] = width
    if base64:
        querystring['base64'] = base64
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-to-screenshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

