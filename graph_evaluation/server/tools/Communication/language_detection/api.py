import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def language_detection(text: str, mode: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detects the language of a piece of text"
    text: The text encoded with UTF-8. A sample of text. Longer text will give a more accurate result, but only the first 1000 characters are considered.
        mode: The response mode desired. Default: "iso2", other values: "json", "xml", "iso3"
        
    """
    url = f"https://langdetect.p.rapidapi.com/language"
    querystring = {'text': text, }
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "langdetect.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

