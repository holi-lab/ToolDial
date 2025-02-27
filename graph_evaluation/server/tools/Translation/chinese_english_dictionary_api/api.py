import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def definition(definition: str, simplified: str=None, limit: int=None, traditional: str=None, any: str=None, exact: bool=None, pinyin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a definition."
    definition: The definition to search for.
        simplified: The simplified Chinese character to search for.
        limit: The maximum number of definitions to return.
        traditional: The traditional Chinese character to search for.
        any: A search term that can be either a definition, simplified Chinese character, traditional Chinese character, or pinyin.
        exact: Flag to set whether to return exact matches only.
        pinyin: The pinyin to search for. This can be either with diacritics or tone numbers.
        
    """
    url = f"https://chinese-english-dictionary-api.p.rapidapi.com/definition"
    querystring = {'definition': definition, }
    if simplified:
        querystring['simplified'] = simplified
    if limit:
        querystring['limit'] = limit
    if traditional:
        querystring['traditional'] = traditional
    if any:
        querystring['any'] = any
    if exact:
        querystring['exact'] = exact
    if pinyin:
        querystring['pinyin'] = pinyin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-english-dictionary-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

