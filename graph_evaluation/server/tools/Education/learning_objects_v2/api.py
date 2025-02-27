import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lang(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the languages currently supported by the API."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/lang"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the types currently supported by the API."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_provider(model: str, keywords: str, lang: str, sort: str, page: int, provider: str, max: int, agemax: int=None, agemin: int=None, learningtimemax: int=None, learningtimemin: int=None, levelmin: int=None, levelmax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of learning objects (LO) that matches specific keywords, language and provider."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/search-provider"
    querystring = {'model': model, 'keywords': keywords, 'lang': lang, 'sort': sort, 'page': page, 'provider': provider, 'max': max, }
    if agemax:
        querystring['ageMax'] = agemax
    if agemin:
        querystring['ageMin'] = agemin
    if learningtimemax:
        querystring['learningTimeMax'] = learningtimemax
    if learningtimemin:
        querystring['learningTimeMin'] = learningtimemin
    if levelmin:
        querystring['levelMin'] = levelmin
    if levelmax:
        querystring['levelMax'] = levelmax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(lang: str, sort: str, type: str, keywords: str, model: str, max: int, page: int, agemin: int=None, bloom: str=None, learningtimemin: int=None, levelmax: int=None, levelmin: int=None, learningtimemax: int=None, popularitymin: int=None, agemax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of learning objects (LO) that matches specific keywords, language and type."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/search"
    querystring = {'lang': lang, 'sort': sort, 'type': type, 'keywords': keywords, 'model': model, 'max': max, 'page': page, }
    if agemin:
        querystring['ageMin'] = agemin
    if bloom:
        querystring['bloom'] = bloom
    if learningtimemin:
        querystring['learningTimeMin'] = learningtimemin
    if levelmax:
        querystring['levelMax'] = levelmax
    if levelmin:
        querystring['levelMin'] = levelmin
    if learningtimemax:
        querystring['learningTimeMax'] = learningtimemax
    if popularitymin:
        querystring['popularityMin'] = popularitymin
    if agemax:
        querystring['ageMax'] = agemax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bloom(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the [Bloom objectives](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy) currently supported by the API."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/bloom"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def provider(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the major providers currently supported by the API."
    
    """
    url = f"https://learning-objects-v2.p.rapidapi.com/provider"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learning-objects-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

