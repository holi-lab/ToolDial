import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_similarity(image_url_2: str, image_url_1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comopute image similarity"
    
    """
    url = f"https://image-similarity-score.p.rapidapi.com/similarity_score"
    querystring = {'image_url_2': image_url_2, 'image_url_1': image_url_1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-similarity-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

