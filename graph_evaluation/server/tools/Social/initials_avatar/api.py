import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def avatar(name: str='john doe', rounded: bool=None, uppercase: bool=None, font_size: int=0, length: int=2, size: int=128, background: str='000000', color: str='ffffff', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates initials avatar"
    name: Name
        rounded: Rounded avatar
        uppercase: Initials uppercase
        font_size: Initials font size between 0.1 to 1 (default 0.5)
        length: Initials length (default 2)
        size: Avatar size between 16 to 256 (default 64)
        background: Background color (RRGGBB)
        color: Initials color (RRGGBB)
        
    """
    url = f"https://initials-avatar.p.rapidapi.com/"
    querystring = {}
    if name:
        querystring['name'] = name
    if rounded:
        querystring['rounded'] = rounded
    if uppercase:
        querystring['uppercase'] = uppercase
    if font_size:
        querystring['font-size'] = font_size
    if length:
        querystring['length'] = length
    if size:
        querystring['size'] = size
    if background:
        querystring['background'] = background
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "initials-avatar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

