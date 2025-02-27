import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def triples_extractor(lang_input: str, text: str='Type or paste here the text you want to analyze', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It extracts triples relations: OBJECT1 RELATION OBJECT2"
    lang_input: Language in wich the text is written. Code: en - english, es - spanish, pt - portuguese and gl - galician
        text: This is the text you are goint to analyze or extract information.
        
    """
    url = f"https://cilenisapi.p.rapidapi.com/triples_extractor"
    querystring = {'lang_input': lang_input, }
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cilenisapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

