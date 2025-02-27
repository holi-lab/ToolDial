import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_one_source_by_name_and_section(name: str, section: str, expire: int=60, language: str='en', defaults: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows source by name and section"
    name: Source name according to registration
        section: Source section name according to registration
        expire: Cache expiration time in minutes
        language: Source language
        defaults: Query source. If defaults is true show queries from default sources
        
    """
    url = f"https://feed-on-news.p.rapidapi.com/sources/{name}/{section}"
    querystring = {}
    if expire:
        querystring['expire'] = expire
    if language:
        querystring['language'] = language
    if defaults:
        querystring['defaults'] = defaults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feed-on-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_sections_by_source(name: str, expire: int=60, defaults: bool=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows all sections from source"
    name: Source name according to registration
        expire: Cache expiration time in minutes
        defaults: Query source. If defaults is true show queries from default sources
        language: Source language
        
    """
    url = f"https://feed-on-news.p.rapidapi.com/sources/{name}"
    querystring = {}
    if expire:
        querystring['expire'] = expire
    if defaults:
        querystring['defaults'] = defaults
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feed-on-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_sources(defaults: bool=None, expire: int=60, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows all standard or user-customized news sources. Both can be filtered by language"
    defaults: Query source. If defaults is true show queries from default sources
        expire: Cache expiration time in minutes
        language: Source language. Example: en, pt-BR.
        
    """
    url = f"https://feed-on-news.p.rapidapi.com/sources"
    querystring = {}
    if defaults:
        querystring['defaults'] = defaults
    if expire:
        querystring['expire'] = expire
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feed-on-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_news(section: str, source: str, expire: int=60, language: str='en', term: str='climate,weather', defaults: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all news from sources
		**Attention! Make request limitation by specifying sources, sections and search terms for it.  To not receive an invalid result by timeout.**"
    section: Source section according to registration
        source: Source name according to registration
        expire: Cache expiration time in minutes
        language: Source language
        term: Query term in news content
        defaults: Query source. If defaults is true show queries from default sources
        
    """
    url = f"https://feed-on-news.p.rapidapi.com/news"
    querystring = {'section': section, 'source': source, }
    if expire:
        querystring['expire'] = expire
    if language:
        querystring['language'] = language
    if term:
        querystring['term'] = term
    if defaults:
        querystring['defaults'] = defaults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feed-on-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

