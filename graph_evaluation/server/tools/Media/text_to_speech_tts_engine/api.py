import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def go(t: str, l: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert textual content to .mp3"
    l: Languages available: ru, en, de, es, pt, fr, nl, zh
        
    """
    url = f"https://text-to-speech-tts-engine.p.rapidapi.com/"
    querystring = {'t': t, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-speech-tts-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

