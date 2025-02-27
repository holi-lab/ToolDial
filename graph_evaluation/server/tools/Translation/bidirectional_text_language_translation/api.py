import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(text: str, lang_to: str, lang_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "takes source text, from language and to language and provides results in a very simple object.
		Languages specified are in two character ISO codes.
		if lang_from is not specified it will be auto detected."
    
    """
    url = f"https://bidirectional-text-language-translation.p.rapidapi.com/translate.php"
    querystring = {'text': text, 'lang_to': lang_to, }
    if lang_from:
        querystring['lang_from'] = lang_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bidirectional-text-language-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_language(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you want to detect language only and not translate you can provide the text parameter and the detected language will be provided in the results"
    
    """
    url = f"https://bidirectional-text-language-translation.p.rapidapi.com/detect.php"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bidirectional-text-language-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supported_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "takes no inputs and provides an array of supported language with iso 2 code and name of language in english."
    
    """
    url = f"https://bidirectional-text-language-translation.p.rapidapi.com/languages.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bidirectional-text-language-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

