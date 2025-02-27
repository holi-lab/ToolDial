import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(text: str, target: str, source: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate"
    text: Text to be translated
        target: Language to be translated
        source: Language of the text source (default: `auto`)
        
    """
    url = f"https://joj-translate.p.rapidapi.com/translate/"
    querystring = {'text': text, 'target': target, }
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-translate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

