import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettranslate(text: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Try quality Translate.
		HTML or Text"
    text: HTML or plain text
        lang: What language to translate into?
Put langfrom-langto (ex. en-fr) OR lang to need translate (ex. fr)

Language flags
[https://rapidapi.com/lebedev.str/api/just-translated/tutorials/language-flags](url)
        
    """
    url = f"https://just-translated.p.rapidapi.com/"
    querystring = {'text': text, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "just-translated.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

