import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_avatar(text: str, size: int=80, fontsize: int=30, color: str='FFFFFF', font: str='sans-serif', background: str='8b75b7', rounded: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get avatar"
    
    """
    url = f"https://avatar-maker1.p.rapidapi.com/avatar"
    querystring = {'text': text, }
    if size:
        querystring['size'] = size
    if fontsize:
        querystring['fontsize'] = fontsize
    if color:
        querystring['color'] = color
    if font:
        querystring['font'] = font
    if background:
        querystring['background'] = background
    if rounded:
        querystring['rounded'] = rounded
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "avatar-maker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

