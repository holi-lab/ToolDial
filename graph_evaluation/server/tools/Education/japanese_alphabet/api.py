import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_language_characters(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- all: characters from hiragana, katakana and combinations
		- full-hiragana: All hiragana characters
		- hiragana: Base hiragana characters
		-hiragana-combinations: hiragana combinations characters
		-full-katakana: All katakana characters
		-katakana: Base katakana characters
		-katakana-combinations: katakana combinations characters"
    
    """
    url = f"https://japanese-alphabet.p.rapidapi.com/v1.0/language/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "japanese-alphabet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

