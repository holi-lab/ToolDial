import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def package(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of mojitok emoticons packages."
    language: Provides a list of mojitok emoticons packages.
        
    """
    url = f"https://mojitok-mojitok-emoticons-v1.p.rapidapi.com/packages"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mojitok-mojitok-emoticons-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emoticons(text: str, language: str='en', page: int=None, list: int=None, extend: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When you type a message, it suggests a list of emoticons to match emotions or situations."
    text: The text you want to convert to an emoticon. For Unicode, must be URL encoded.
        language: Sets the language for the returned result value. It provides en(English), ja(Japanese), zh(Chinese), vi(Vietnamese) andko(Korean), and is treated as ko if no value is specified or if it is not correct.
        page: This parameter is for pagination
        list: Supports up to 1000 list per page.
        extend: Shows additional information on emoticons.
        
    """
    url = f"https://mojitok-mojitok-emoticons-v1.p.rapidapi.com/emoticons"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    if list:
        querystring['list'] = list
    if extend:
        querystring['extend'] = extend
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mojitok-mojitok-emoticons-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

