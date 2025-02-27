import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2_thumbnail_ready(size: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Poll this API until it returns that the image is available.  If available, you can download through endpoint 1.  Rendering an image depends largely on how fast a particular web page is loaded (and if it contains Flash).  Average waiting time is around 20 - 60 seconds"
    size: t = Tiny, 90 x 68 pixels;  s= Small, 120 x 90 pixels;  m = Medium, 200 x 150 pixels; l = Large, 400 x 300 pixels;  x = Extra large, 480 x 360 pixels
        url: The URL to generate the thumbnail from
        
    """
    url = f"https://pagepeeker-pagepeeker.p.rapidapi.com/thumbs_ready.php"
    querystring = {'size': size, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepeeker-pagepeeker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_shoot_thumbnail(size: str, url: str, refresh: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Issues a reset API call if refresh is set to 1. Else it will download if the image is ready (see endpoint 2.)"
    size: t = Tiny, 90 x 68 pixels;  s= Small, 120 x 90 pixels;  m = Medium, 200 x 150 pixels; l = Large, 400 x 300 pixels;  x = Extra large, 480 x 360 pixels
        url: The URL to generate the thumbnail from
        refresh: This parameter forces the currently generate d thumbnail to be regenerated.  It is optional and will be ignored unless it contains the value 1
        
    """
    url = f"https://pagepeeker-pagepeeker.p.rapidapi.com/thumbs.php"
    querystring = {'size': size, 'url': url, }
    if refresh:
        querystring['refresh'] = refresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepeeker-pagepeeker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

