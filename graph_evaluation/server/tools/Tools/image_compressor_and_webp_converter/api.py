import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_compressor(type: str, quality: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image Compression tool that will compress the Image, convert it to WebP format and return the image in response of request. 
		You need to pass following parameters,
		1. url/file = either URL of image or file as attachment.
		2. type =  lossy or lossless.
		3. quality = quality of image."
    
    """
    url = f"https://image-compressor-and-webp-converter.p.rapidapi.com/Compressor"
    querystring = {'type': type, 'quality': quality, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-compressor-and-webp-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

