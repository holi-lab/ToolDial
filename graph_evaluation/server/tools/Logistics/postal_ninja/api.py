import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listcarriers(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of all supported carriers."
    lang: Language code
        
    """
    url = f"https://postal-ninja.p.rapidapi.com/v1/carriers"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postal-ninja.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtracks(max: str, events: str, offset: str, lang: str, trackcodes: str=None, ids: str=None, list: str='ACTIVE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all or selected packages with pagination."
    max: List size
        events: Include or not events info for each selected package
        offset: List offset
        lang: Language to translate event descriptions and locations to: AS_IS - no translation at all, EN - use only stored translations to English, EN_MT - use machine translation by Google if there is no stored translation for this term, etc.
        trackcodes: Return packages only with tracking numbers from given comma-separated list
        ids: Return packages only with IDs from given comma-separated list
        list: Select packages only from given list. Or any - if not specified
        
    """
    url = f"https://postal-ninja.p.rapidapi.com/v1/track"
    querystring = {'max': max, 'events': events, 'offset': offset, 'lang': lang, }
    if trackcodes:
        querystring['trackCodes'] = trackcodes
    if ids:
        querystring['ids'] = ids
    if list:
        querystring['list'] = list
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postal-ninja.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns OK for authorized API user"
    
    """
    url = f"https://postal-ninja.p.rapidapi.com/v1/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postal-ninja.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettrack(is_id: str, await: bool, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns tracking info for given package ID."
    id: Package ID
        await: If true - await previously submitted refresh request to complete, if false - don't wait and return package info immediately
        lang: Language to translate event descriptions and locations to: AS_IS - no translation at all, EN - use only stored translations to English, EN_MT - use machine translation by Google if there is no stored translation for this term, etc.
        
    """
    url = f"https://postal-ninja.p.rapidapi.com/v1/track/{is_id}"
    querystring = {'await': await, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "postal-ninja.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

