import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_language(text: str, detectedcount: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Feed this API a few sentences and have it determine what language it is with a confidence score"
    
    """
    url = f"https://quick-language-detector.p.rapidapi.com/v1/detectlanguage"
    querystring = {'text': text, }
    if detectedcount:
        querystring['detectedcount'] = detectedcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quick-language-detector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

