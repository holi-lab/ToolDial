import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def color_to_picture(height: str='200', color: str='ff0000', width: str='200', mode: str='RGB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes prompt of picture mode (L,RGB,RGBA), color in hex (example: ff03bc), width (example: 200), height (example: 200) and makes new image in single color)"
    
    """
    url = f"https://color-to-picture-api.p.rapidapi.com/color-image"
    querystring = {}
    if height:
        querystring['height'] = height
    if color:
        querystring['color'] = color
    if width:
        querystring['width'] = width
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "color-to-picture-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

