import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup(to: str, word: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get bilingual translations"
    to: The language you want to get translations
        word: The word you want to translate
        from: The language that your word belongs
        
    """
    url = f"https://link-bilingual-dictionary.p.rapidapi.com/{is_from}/{to}/{word}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "link-bilingual-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

