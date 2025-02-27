import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gesture_recognition_api(image: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A gesture recognition API can be used in a wide range of applications, from human-computer interaction (such as controlling a system through gestures) to sign language interpretation and beyond."
    
    """
    url = f"https://gesture-recognition-api1.p.rapidapi.com/coc"
    querystring = {}
    if image:
        querystring['Image'] = image
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gesture-recognition-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gesture_recognition_api(image: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A gesture recognition API can be used in a wide range of applications, from human-computer interaction (such as controlling a system through gestures) to sign language interpretation and beyond."
    
    """
    url = f"https://gesture-recognition-api1.p.rapidapi.com/coc"
    querystring = {}
    if image:
        querystring['Image'] = image
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gesture-recognition-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

