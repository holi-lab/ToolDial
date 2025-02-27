import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_base(url: str, name: str='image_name', save: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This return the image for web"
    
    """
    url = f"https://qr-code54.p.rapidapi.com/qr-code"
    querystring = {'url': url, }
    if name:
        querystring['name'] = name
    if save:
        querystring['save'] = save
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

