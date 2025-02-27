import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def img_rezize_url(url: str, h: str, w: str, t: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "test"
    h: Height
        w: Width
        t: object-fit:
\"cover\" | \"contain\" | \"fill\" | \"none\" | \"scale-down\"
        
    """
    url = f"https://to-image1.p.rapidapi.com/img"
    querystring = {'url': url, 'h': h, 'w': w, }
    if t:
        querystring['t'] = t
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "to-image1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

