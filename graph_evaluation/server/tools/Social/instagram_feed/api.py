import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feed(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get feed."
    
    """
    url = f"https://instagram-feed2.p.rapidapi.com/?action=feed"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-feed2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image(url: str='https://instagram.falg5-1.fna.fbcdn.net/v/t51.2885-15/e35/c135.0.810.810a/s150x150/42905762_330257041065667_6035130151857749848_n.jpg?tp=1&_nc_ht=instagram.falg5-1.fna.fbcdn.net&_nc_cat=107&_nc_ohc=3b__PReg750AX_f5H0V&edm=ABZsPhsBAAAA&ccb=7-4&oh=2148c58fe511971adb246ead813aacfa&oe=60C98D5C&_nc_sid=4efc9f', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image"
    
    """
    url = f"https://instagram-feed2.p.rapidapi.com/?action=image"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-feed2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

