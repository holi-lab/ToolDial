import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(text: str, tolang: str, fromlang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translates input text, returning translated text"
    text: The input text to translate.
        tolang: The language to use for translation of the input text, set to one of the language codes listed in the overview tab
        fromlang: The language of the fromLang, set to one of the language codes listed in Language Support. If the source language is not specified, the API will attempt to detect the source language automatically and return it within the response.
        
    """
    url = f"https://translate-all-languages.p.rapidapi.com/translate"
    querystring = {'text': text, 'toLang': tolang, }
    if fromlang:
        querystring['fromLang'] = fromlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate-all-languages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of supported languages for translation."
    
    """
    url = f"https://translate-all-languages.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate-all-languages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detects the language of text within a request."
    text: The input text upon which to perform language detection. Repeat this parameter to perform language detection on multiple text inputs.
        
    """
    url = f"https://translate-all-languages.p.rapidapi.com/detect"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "translate-all-languages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

