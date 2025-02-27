import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def checks_if_an_image_contains_nudity(url: str='https://live.staticflickr.com/2567/3675799660_9eb0911619_o_d.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns two fields: "classification" is either "safe" or "nudity", and "confidence" returns a confidence level between 0 and 1. An image is classified into the "nudity" category if a person's sexual organs, breast or belly are displayed."
    url: Url of the image to analyze
        
    """
    url = f"https://nsfw-nudity-image-classifier.p.rapidapi.com/check_image"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nsfw-nudity-image-classifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

