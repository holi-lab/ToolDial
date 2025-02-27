import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dzi_to_image(url: str, resizewidth: int=None, format: str='jpeg', resizeheight: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://dzi-stitcher.p.rapidapi.com/dzi"
    querystring = {'url': url, }
    if resizewidth:
        querystring['resizeWidth'] = resizewidth
    if format:
        querystring['format'] = format
    if resizeheight:
        querystring['resizeHeight'] = resizeheight
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dzi-stitcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

