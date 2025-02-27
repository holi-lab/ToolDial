import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate_using_query(target_lang: str='de', text: str='Hello world!', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate with GET method. 
		Use text query parameter to specify text that should be translated.
		Use target_lang to specify target language."
    
    """
    url = f"https://amazon-translate.p.rapidapi.com/"
    querystring = {}
    if target_lang:
        querystring['target_lang'] = target_lang
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-translate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

