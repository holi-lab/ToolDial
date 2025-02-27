import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_to_speech(lang: str, speed: str='0.5', text: str='How are you?', ssml: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Text to Speech Voice Reader"
    lang: Speech Language
        speed: between `0.1` and `1.0`
        text: Text
        ssml: SSML

*At least one of the `text` or `ssml` parameters is required.*
        
    """
    url = f"https://text-to-speech48.p.rapidapi.com/synthesize"
    querystring = {'lang': lang, }
    if speed:
        querystring['speed'] = speed
    if text:
        querystring['text'] = text
    if ssml:
        querystring['ssml'] = ssml
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-speech48.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

