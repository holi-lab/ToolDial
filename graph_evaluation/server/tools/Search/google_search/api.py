import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def complete_web_search(q: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Includes full functionality"
    q: Query
        language: The language with which the search will be carried out and will be returned
        
    """
    url = f"https://google-search90.p.rapidapi.com/completesearch"
    querystring = {'q': q, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search90.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lite_web_search(language: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The limited version of the web search, for free users"
    language: The language with which the search will be carried out and will be returned
        
    """
    url = f"https://google-search90.p.rapidapi.com/litesearch"
    querystring = {'language': language, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search90.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_search(language: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Web search with most features, these remain excluded: instant results (but there is WikiBox (data from Wikipedia) from which you will be able to extract them) and music"
    language: The language with which the search will be carried out and will be returned
        q: Query
        
    """
    url = f"https://google-search90.p.rapidapi.com/search"
    querystring = {'language': language, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search90.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

