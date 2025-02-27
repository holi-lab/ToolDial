import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tester_of_headers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test secret headers and parameters"
    
    """
    url = f"https://lenguaje.p.rapidapi.com/tester"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lenguaje.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_de_conjugaci_n_de_verbos(in_word: str, nested: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve la conjugación de un verbo en infinitivo"
    
    """
    url = f"https://lenguaje.p.rapidapi.com/conjugator02"
    querystring = {'in_word': in_word, }
    if nested:
        querystring['nested'] = nested
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lenguaje.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarity_spanish(max_suggestions: int, in_word: str, distance: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain Spanish words that are spelt or sound similar to the input word."
    
    """
    url = f"https://lenguaje.p.rapidapi.com/similars02"
    querystring = {'max_suggestions': max_suggestions, 'in_word': in_word, 'distance': distance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lenguaje.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lemmatizer_spanish(in_word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain the lemmas of an input word in Spanish."
    
    """
    url = f"https://lenguaje.p.rapidapi.com/lemmatizer02"
    querystring = {'in_word': in_word, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lenguaje.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

