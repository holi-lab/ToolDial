import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_text_to_speech_via_http_get(src: str, hl: str, r: int=0, c: str='mp3', f: str='8khz_8bit_mono', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts textual content to audio content"
    src: The textual content for converting to speech
        hl: The textual content language
        r: The speech rate (speed). Allows values: from -10 (slowest speed) up to 10 (fastest speed). Default value: 0 (normal speed)
        c: The speech audio codec
        f: The speech audio formats
        
    """
    url = f"https://voicerss-text-to-speech.p.rapidapi.com/"
    querystring = {'src': src, 'hl': hl, }
    if r:
        querystring['r'] = r
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "voicerss-text-to-speech.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

