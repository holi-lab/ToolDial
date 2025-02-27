import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_lookup_word_word(word: str, synonyms: bool=None, antonyms: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the synonyms and antonyms of the specified word."
    word: Le mot à rechercher
        synonyms: Inclure des synonymes dans la réponse
        antonyms: Inclure des antonymes dans la réponse
        
    """
    url = f"https://synonymes-et-antonymes1.p.rapidapi.com/api/lookup_word/{word}"
    querystring = {}
    if synonyms:
        querystring['synonyms'] = synonyms
    if antonyms:
        querystring['antonyms'] = antonyms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synonymes-et-antonymes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

