import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def screenshot(height: str, width: str, url: str, fullscreen: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Take a screenshot from a webpage url."
    
    """
    url = f"https://website-screenshot6.p.rapidapi.com/screenshot"
    querystring = {'height': height, 'width': width, 'url': url, }
    if fullscreen:
        querystring['fullscreen'] = fullscreen
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-screenshot6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

