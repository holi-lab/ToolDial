import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_face(accuracy_boost: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use our face recognition API to recognize the position of human faces in your images.""
    
    """
    url = f"https://deep-face-detection.p.rapidapi.com/face2/"
    querystring = {'accuracy_boost': accuracy_boost, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deep-face-detection.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recognize_face_w_age_gender(accuracy_boost: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use our face recognition API to recognize the position of human faces in your images along with their predicted age and gender."
    
    """
    url = f"https://deep-face-detection.p.rapidapi.com/face1/"
    querystring = {'accuracy_boost': accuracy_boost, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deep-face-detection.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

