import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_a_random_word(api_key: str='5w36eV0FZJu9QIPlpR18', generator: str='words', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scour through thousands of words from the RandomWordGenerator.com database"
    
    """
    url = f"https://random-word-generator2.p.rapidapi.com/word.php"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    if generator:
        querystring['generator'] = generator
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_a_random_sentence(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random sentence"
    
    """
    url = f"https://random-word-generator2.p.rapidapi.com/sentence.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

