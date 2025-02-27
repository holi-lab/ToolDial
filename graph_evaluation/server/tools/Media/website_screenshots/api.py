import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(width: int, url: str, format: str=None, delay: int=2500, mobile: bool=None, fullpage: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates a thumbnail for the given url."
    width: The width of the screenshot (32 .. 1024).
        url: The URL of the website you want to caputre. E.g. http://thumbnail.ws/ or https://www.google.com/.
        format: The output format. Currently PNG and JPEG are supported.
        delay: How long to wait after the page has loaded to create a screenshot. Time in millisecond. Default is 2500.
        mobile: If set to true a screenshot of the mobile website will be returned.
        fullpage: If set to true, the entire page will be rendered.
        
    """
    url = f"https://thumbnail-thumbnail-v1.p.rapidapi.com/get"
    querystring = {'width': width, 'url': url, }
    if format:
        querystring['format'] = format
    if delay:
        querystring['delay'] = delay
    if mobile:
        querystring['mobile'] = mobile
    if fullpage:
        querystring['fullpage'] = fullpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thumbnail-thumbnail-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

