import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for images."
    keyword: A keyword that describes the sought after picture.
        
    """
    url = f"https://webknox-search.p.rapidapi.com/media/images/search"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_search(query: str, language: str='en', number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Web pages containing certain query terms."
    query: The query to the index.
        language: The language (in ISO 6391, e.g. 'en' = English.
        number: The number of results to retrieve [1,50].
        
    """
    url = f"https://webknox-search.p.rapidapi.com/webpage/search"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

