import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def adult_image_detection(callback: str=None, url: str='https://www.moderatecontent.com/img/sample_face_6.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect adult image"
    callback: Use this for a jsonp style call
        url: The url of the image you would like classify
        
    """
    url = f"https://mdr8.p.rapidapi.com/api/v2"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdr8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

