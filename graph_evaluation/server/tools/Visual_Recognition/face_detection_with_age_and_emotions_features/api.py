import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def face_detection_with_age_gender_emotions_facial_features(image_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Use our Face Detection API to detect the location of human faces in your images along with their predicted ages, genders, emotions & facial features**
		
		*Image must be a regular JPEG or PNG image. Usually such images have extensions: .jpg, .jpeg, .png. The service checks input file by MIME type …"
    
    """
    url = f"https://face-detection-with-age-and-emotions-features.p.rapidapi.com/all"
    querystring = {'image_url': image_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "face-detection-with-age-and-emotions-features.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def face_detection(image_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Use our Face Detection API to detect the location of human faces in your images.**
		
		*Image must be a regular JPEG or PNG image. Usually such images have extensions: .jpg, .jpeg, .png. The service checks input file by MIME type …"
    
    """
    url = f"https://face-detection-with-age-and-emotions-features.p.rapidapi.com/"
    querystring = {'image_url': image_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "face-detection-with-age-and-emotions-features.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

