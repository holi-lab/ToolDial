import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def chatgpt_detector_api(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our API can detect text in over 140 languages"
    
    """
    url = f"https://chatgpt-detector-ai-powered-chatgpt-answer-detector.p.rapidapi.com/chatgptDetector"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chatgpt-detector-ai-powered-chatgpt-answer-detector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

