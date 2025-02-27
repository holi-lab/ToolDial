import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(to: str, text: str, is_from: str='en', protected_words: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate Text or HTML"
    
    """
    url = f"https://nlp-translation.p.rapidapi.com/v1/translate"
    querystring = {'to': to, 'text': text, }
    if is_from:
        querystring['from'] = is_from
    if protected_words:
        querystring['protected_words'] = protected_words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nlp-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

