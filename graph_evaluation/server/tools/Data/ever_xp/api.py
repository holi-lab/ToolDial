import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def humanitarian_support_charity_organizations(humanitarian: int, api_key: str, voice: int=1, max_l: int=9999, freshness: int=1, pull: int=1, style: int=1, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Headings By Humanitarian Categories"
    
    """
    url = f"https://ever-xp.p.rapidapi.com/heading/humanitarian"
    querystring = {'humanitarian': humanitarian, 'api_key': api_key, }
    if voice:
        querystring['voice'] = voice
    if max_l:
        querystring['max_l'] = max_l
    if freshness:
        querystring['freshness'] = freshness
    if pull:
        querystring['pull'] = pull
    if style:
        querystring['style'] = style
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ever-xp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dates_times(lang: str, api_key: str, time: int, style: int=1, freshness: int=1, min_l: int=1, voice: int=1, max_l: int=9999, pull: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Headings by date and time categories"
    
    """
    url = f"https://ever-xp.p.rapidapi.com/heading/time"
    querystring = {'lang': lang, 'api_key': api_key, 'time': time, }
    if style:
        querystring['style'] = style
    if freshness:
        querystring['freshness'] = freshness
    if min_l:
        querystring['min_l'] = min_l
    if voice:
        querystring['voice'] = voice
    if max_l:
        querystring['max_l'] = max_l
    if pull:
        querystring['pull'] = pull
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ever-xp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def patterns(lang: str, pattern: int, api_key: str, max_l: int=99999, freshness: int=1, voice: int=1, pull: int=1, min_l: int=1, style: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Headings By Pattern Id"
    
    """
    url = f"https://ever-xp.p.rapidapi.com/heading/pattern"
    querystring = {'lang': lang, 'pattern': pattern, 'api_key': api_key, }
    if max_l:
        querystring['max_l'] = max_l
    if freshness:
        querystring['freshness'] = freshness
    if voice:
        querystring['voice'] = voice
    if pull:
        querystring['pull'] = pull
    if min_l:
        querystring['min_l'] = min_l
    if style:
        querystring['style'] = style
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ever-xp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

