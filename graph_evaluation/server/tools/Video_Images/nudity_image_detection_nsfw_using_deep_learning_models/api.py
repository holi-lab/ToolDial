import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nudity_image_detection(img: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nudity Image Detection"
    img: Image Url for detection
        
    """
    url = f"https://nudity-image-detection-nsfw-using-deep-learning-models.p.rapidapi.com/nsfw"
    querystring = {'img': img, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nudity-image-detection-nsfw-using-deep-learning-models.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

