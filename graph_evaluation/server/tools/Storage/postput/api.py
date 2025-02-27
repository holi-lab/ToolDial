import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transform_your_images(grayscale: bool=None, rotate: int=90, blur: int=4, resize: str='100,100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform on-the-fly operations on your files"
    
    """
    url = f"https://postput.p.rapidapi.com/6bcc4d12-a024-48be-a067-7c9605cc8397.png"
    querystring = {}
    if grayscale:
        querystring['grayscale'] = grayscale
    if rotate:
        querystring['rotate'] = rotate
    if blur:
        querystring['blur'] = blur
    if resize:
        querystring['resize'] = resize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postput.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transform_your_images(grayscale: bool=None, rotate: int=90, blur: int=4, resize: str='100,100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform on-the-fly operations on your files"
    
    """
    url = f"https://postput.p.rapidapi.com/6bcc4d12-a024-48be-a067-7c9605cc8397.png"
    querystring = {}
    if grayscale:
        querystring['grayscale'] = grayscale
    if rotate:
        querystring['rotate'] = rotate
    if blur:
        querystring['blur'] = blur
    if resize:
        querystring['resize'] = resize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postput.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

