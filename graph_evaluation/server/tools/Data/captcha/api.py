import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def solve(image: str, timeout: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Solve a Captcha given an image URL"
    image: HTTP Path/URL of the Captcha Image
        timeout: Timeout in seconds (Default = 20 seconds)
        
    """
    url = f"https://metropolis-api-captcha.p.rapidapi.com/solve"
    querystring = {'image': image, }
    if timeout:
        querystring['timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-captcha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

