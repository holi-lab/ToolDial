import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_zodiac(sign: str, lang: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return specific zodiac fortune by sign, period and language."
    lang: Chinese Simplified or Traditional
        
    """
    url = f"https://chinese-zodiacs-ai.p.rapidapi.com/get_zodiac/{sign}/{period}/{lang}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-zodiacs-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

