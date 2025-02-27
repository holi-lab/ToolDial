import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(text: str, target_lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://cloud.google.com/translate/docs/languages"
    target_lang: enter language code of target language ( you can get language code here : https://cloud.google.com/translate/docs/languages)
        
    """
    url = f"https://google-translate112.p.rapidapi.com/translate"
    querystring = {'text': text, 'target_lang': target_lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-translate112.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

