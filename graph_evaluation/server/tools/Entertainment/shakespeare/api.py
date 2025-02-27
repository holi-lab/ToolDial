import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shakespeare_generate_lorem_ipsum(limit: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Shakespeare lorem ipsum."
    limit: No of elements to generate
        type: Type of element to generate `paragraphs/sentences/words`.
        
    """
    url = f"https://shakespeare1.p.rapidapi.com/shakespeare/generate/lorem-ipsum"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shakespeare_generate_name(limit: int=None, variation: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate random Shakespearen names."
    limit: No of names to generate
        variation: Variation to generate `male/female`.
        
    """
    url = f"https://shakespeare1.p.rapidapi.com/shakespeare/generate/name"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if variation:
        querystring['variation'] = variation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shakespeare_translate(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate from English to Shakespeare English."
    text: Text to translate to Shakespeare English.
        
    """
    url = f"https://shakespeare1.p.rapidapi.com/shakespeare/translate"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shakespeare_generate_insult(limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate random Shakespeare style insults."
    limit: No of insults to generate
        
    """
    url = f"https://shakespeare1.p.rapidapi.com/shakespeare/generate/insult"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shakespeare_quote(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random Shakespeare quote."
    
    """
    url = f"https://shakespeare1.p.rapidapi.com/shakespeare/quote"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

