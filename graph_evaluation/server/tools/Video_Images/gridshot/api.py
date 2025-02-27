import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def caputre(viewport_height: int, url: str, viewport_width: int, fullpage: bool=None, javascriptenabled: bool=None, height: int=600, width: int=600, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Capture the screenshot by passing URL
		
		Other parameters you can pass:
		
		-  viewport_width:  [Default: 1920]
		- viewport_height: [Default: 1080]
		- javascriptEnabled: [Default: false]
		- fullPage: [Default: true]
		- width: [Default: 600]
		- height: [Default: 600]"
    
    """
    url = f"https://gridshot.p.rapidapi.com/screenshot"
    querystring = {'viewport_height': viewport_height, 'url': url, 'viewport_width': viewport_width, }
    if fullpage:
        querystring['fullPage'] = fullpage
    if javascriptenabled:
        querystring['javascriptEnabled'] = javascriptenabled
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gridshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

