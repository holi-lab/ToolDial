import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_joke(categories: str, contains: str='C%23', type: str=None, idrange: str='0-150', blacklistflags: str='nsfw,racist', format: str='json', safe_mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the endpoint you need to call to get a joke."
    categories: A single category or comma-separated list of categories or the value Any to get jokes of a random category
        contains: If specified, only jokes that contain this string will be served. Value needs to be percent-encoded.
        type: Will make JokeAPI only serve jokes with this joke type. Leave empty to randomly select a type.
        idrange: A single ID or range of IDs to serve jokes from
        blacklistflags: A single or comma-separated list of blacklist flags
        format: The file format the joke should be served with
        safe_mode: A value-less parameter, that, if present, will tell JokeAPI to try its hardest not to serve any offensive jokes. By default, every joke that has one or more flags set to true is considered unsafe, in addition to a few more hand-picked ones.
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/joke/{categories}"
    querystring = {}
    if contains:
        querystring['contains'] = contains
    if type:
        querystring['type'] = type
    if idrange:
        querystring['idRange'] = idrange
    if blacklistflags:
        querystring['blacklistFlags'] = blacklistflags
    if format:
        querystring['format'] = format
    if safe_mode:
        querystring['safe-mode'] = safe_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides some information on JokeAPI"
    format: The file format the joke should be served with
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/info"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list / an array of all available joke categories"
    format: The file format the joke should be served with. More info: https://sv443.net/jokeapi/v2/#response-formats
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/categories"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is intended for external uptime monitoring but you can also use it if you want to."
    format: The file format the joke should be served with. More info: https://sv443.net/jokeapi/v2/#response-formats
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/ping"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flags(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list / an array of all available blacklist flags"
    format: The file format the joke should be served with. More info: https://sv443.net/jokeapi/v2/#response-formats
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/flags"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def formats(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list / an array of all available response formats"
    format: The file format the joke should be served with. More info: https://sv443.net/jokeapi/v2/#response-formats
        
    """
    url = f"https://jokeapi-v2.p.rapidapi.com/formats"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

