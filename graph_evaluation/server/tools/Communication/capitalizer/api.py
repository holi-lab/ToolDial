import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def capitalizer(text: str, lang: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Changes the capitalization of incoming text based on language rules of that text and responds with a simple string"
    text: The text to be capitalized
        lang: The ISO 639-1 code if know or "autoDetect" to detect language before applying the change.
        mode: The target capitalization mode: default, title, sentence, lower, upper, nospace
        
    """
    url = f"https://sprawkcapitalizer.p.rapidapi.com/api/applyCaps"
    querystring = {'text': text, 'lang': lang, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sprawkcapitalizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

