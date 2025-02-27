import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def obfuscate(word: str, level: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes a word as parameter obfuscates it using a proprietary Unicode Character codex"
    word: The word to be obfuscated
        level: 1 - 6, 6 being most obfuscated
        
    """
    url = f"https://bheithaus-unicode-obfuscator.p.rapidapi.com/obfuscate"
    querystring = {'word': word, }
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bheithaus-unicode-obfuscator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

