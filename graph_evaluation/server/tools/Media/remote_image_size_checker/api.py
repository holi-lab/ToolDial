import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check(width: int, height: int, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify if a remote image satisfy a required minimum size."
    width: Minimum Width
        height: Minimum Height
        url: Remote image URL
        
    """
    url = f"https://tgeselle-remote-image-size-checker-v1.p.rapidapi.com/check"
    querystring = {'width': width, 'height': height, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tgeselle-remote-image-size-checker-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

