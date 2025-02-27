import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(language: str, word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We use the Get method in our api, which returns a string with the synonyms separated by commas, which can later be treated in the programming language you use.
		
		Word - here you put the word you want to know the synonym of.
		Language - EN, ES, FR and PT representing English, Spanish, French and Portuguese, just put one of the representations .
		
		If the word has no synonym, the return is the word itself."
    
    """
    url = f"https://synonyms-words.p.rapidapi.com/"
    querystring = {'language': language, 'word': word, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synonyms-words.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

