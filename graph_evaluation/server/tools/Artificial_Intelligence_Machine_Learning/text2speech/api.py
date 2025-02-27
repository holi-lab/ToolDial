import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text2speech_endpoint(text: str, lang: str=None, slow: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/tts"
    
    """
    url = f"https://text2speech3.p.rapidapi.com/tts"
    querystring = {'text': text, }
    if lang:
        querystring['lang'] = lang
    if slow:
        querystring['slow'] = slow
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text2speech3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

