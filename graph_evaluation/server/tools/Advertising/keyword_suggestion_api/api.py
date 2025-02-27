import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all(keyword: str, country_code: str='us', language_code: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you all keywords releated to your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/all"
    querystring = {'keyword': keyword, }
    if country_code:
        querystring['country_code'] = country_code
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alphabetical(keyword: str, country_code: str='us', language_code: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you alphabetical releated to your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/alphabetical"
    querystring = {'keyword': keyword, }
    if country_code:
        querystring['country_code'] = country_code
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comparisons(keyword: str, country_code: str='us', language_code: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you comparisons releated to your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/comparisons"
    querystring = {'keyword': keyword, }
    if country_code:
        querystring['country_code'] = country_code
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preposition(keyword: str, language_code: str='en', country_code: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you preposition releated to your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/preposition"
    querystring = {'keyword': keyword, }
    if language_code:
        querystring['language_code'] = language_code
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def questions(keyword: str, country_code: str='us', language_code: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you questions releated to your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/questions"
    querystring = {'keyword': keyword, }
    if country_code:
        querystring['country_code'] = country_code
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def releated(keyword: str, language_code: str='en', country_code: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will give you releated keywords for your queried keyword"
    
    """
    url = f"https://keyword-suggestion-api.p.rapidapi.com/related"
    querystring = {'keyword': keyword, }
    if language_code:
        querystring['language_code'] = language_code
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-suggestion-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

