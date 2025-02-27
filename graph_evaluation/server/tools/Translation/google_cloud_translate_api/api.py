import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate_using_query(text: str='Hello world!', target_lang: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate with GET method. 
		Use text query parameter to specify text that should be translated.
		Use target_lang to specify target language."
    
    """
    url = f"https://google-cloud-translate-api.p.rapidapi.com/"
    querystring = {}
    if text:
        querystring['text'] = text
    if target_lang:
        querystring['target_lang'] = target_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-cloud-translate-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

