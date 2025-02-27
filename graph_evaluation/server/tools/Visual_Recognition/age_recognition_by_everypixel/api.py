import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def age_recognition(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Age Recognition API detects an age of people on photos. Model is trained on few datasets up to 300 000 images of people of different gender, age and nationality."
    
    """
    url = f"https://age-recognition-by-everypixel.p.rapidapi.com/age_recognition"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "age-recognition-by-everypixel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

