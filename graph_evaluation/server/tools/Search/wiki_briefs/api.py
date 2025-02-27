import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_wikihow(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search any 'how-to' question on WikiHow and returns summarized methods and steps."
    q: Any how-to question to ask. For example, 'find meaning in life', 'learn any language', 'play soccer', 'change an oil filter'...
        
    """
    url = f"https://wiki-briefs.p.rapidapi.com/howto"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wiki-briefs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_wikipedia(q: str, topk: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search anything on Wikipedia and returns top K summarized information and similar items. Images and coordinates with map link are provided when available."
    q: Query string, can be anything.
        topk: Return top k summarized information. Default to 5. Must be greater than 0.
        
    """
    url = f"https://wiki-briefs.p.rapidapi.com/search"
    querystring = {'q': q, }
    if topk:
        querystring['topk'] = topk
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wiki-briefs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

