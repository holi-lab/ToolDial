import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crop(url: str, focus: str='1200x300', edge: int=None, zoom: int=None, format: str=None, size: str='600x600', quality: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Crop Image**
		
		- url: string - Your image URL eg  http://domain.tld/myimage.jpg
		- size: (int)x(int) - Crop size like 600x600 or 1200x400
		- focus: (int)x(int) - Focus point (Image Subject) like 650x320
		- zoom: int - Zoom factor from 0  to 99 
		- edge: float - Subject target position. 2.0  center if possible. Greater 2.0 ad margin around the  subject.
		- format: string - jpeg|webp|png
		- quality: int - Compression quality from 0-100"
    
    """
    url = f"https://image-cropper.p.rapidapi.com/crop"
    querystring = {'url': url, }
    if focus:
        querystring['focus'] = focus
    if edge:
        querystring['edge'] = edge
    if zoom:
        querystring['zoom'] = zoom
    if format:
        querystring['format'] = format
    if size:
        querystring['size'] = size
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-cropper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

