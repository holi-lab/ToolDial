import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_random_words(array_size: str, language: str, word_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates A List Of Random Words."
    
    """
    url = f"https://random-words-spanish-and-french.p.rapidapi.com/{language}/array/{array_size}/{word-size}"
    querystring = {}
    if word_size:
        querystring['word-size'] = word_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-words-spanish-and-french.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_word(language: str, word_size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get One Random Word"
    
    """
    url = f"https://random-words-spanish-and-french.p.rapidapi.com/{language}/one-word/{word-size}"
    querystring = {}
    if word_size:
        querystring['word-size'] = word_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-words-spanish-and-french.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

