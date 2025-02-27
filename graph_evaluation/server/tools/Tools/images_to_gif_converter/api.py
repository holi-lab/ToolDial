import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def converttogif(imageurls: str, delay: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert list of images to a GIF image"
    imageurls: Comma separated URLs of images which needs to be converted into a GIF
        delay: How much should be delay between the animation of 2 images. It will be in milliseconds. Default is 1000 (i.e 1 second)
        
    """
    url = f"https://udayogra-images-to-gif-converter-v1.p.rapidapi.com/am"
    querystring = {'imageurls': imageurls, 'delay': delay, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "udayogra-images-to-gif-converter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

