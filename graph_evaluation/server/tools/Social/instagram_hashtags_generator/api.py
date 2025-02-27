import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_image_based_hashtags(image_url: str, max_labels: int=15, min_confidence: int=85, name: str='New York image', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a comprehensive list of hashtags based on the image you are sending.
		
		Image must be a regular JPEG or PNG image. Usually such images have extensions: .jpg, .jpeg, .png. The service checks input file by MIME type ..."
    image_url: The image url you want to get hashtags for
        max_labels: Max number of hashtags to detect
        min_confidence: confidence threshold
        name: The image name
        
    """
    url = f"https://instagram-hashtags-generator.p.rapidapi.com/"
    querystring = {'image_url': image_url, }
    if max_labels:
        querystring['max_labels'] = max_labels
    if min_confidence:
        querystring['min_confidence'] = min_confidence
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-hashtags-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

