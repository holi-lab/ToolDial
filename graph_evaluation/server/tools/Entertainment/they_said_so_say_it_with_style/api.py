import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qod(language: str='en', category: str='inspire', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quote of the day"
    category: Category of the desired QOD.
        
    """
    url = f"https://theysaidso.p.rapidapi.com/qod"
    querystring = {}
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_authors(language: str='en', limit: int=5, start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of popular authors available in the system"
    
    """
    url = f"https://theysaidso.p.rapidapi.com/quote/authors/popular"
    querystring = {}
    if language:
        querystring['language'] = language
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_categories(limit: int=5, start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of popular Categories available in the system"
    
    """
    url = f"https://theysaidso.p.rapidapi.com/quote/categories/popular"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qod_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the categories for QOD"
    
    """
    url = f"https://theysaidso.p.rapidapi.com/qod/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_quote(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random quote."
    
    """
    url = f"https://theysaidso.p.rapidapi.com/quote/random"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_authors(query: str, language: str='en', limit: int=5, start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the list of  authors available in the system"
    
    """
    url = f"https://theysaidso.p.rapidapi.com/quote/authors/search"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quote(query: str, maxlength: int=500, minlength: int=100, category: str=None, author: str='Steve Jobs', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the quotes database based on author, tag etc."
    query: Query String to query. These words will be searched in the quote and the matched quotes will be returned.
        maxlength: Maximum length of resulting quote
        minlength: Minimum required length of the resulting quote
        category: Category(tag) from which you want the resulting quote from
        author: Random quote from the given author
        
    """
    url = f"https://theysaidso.p.rapidapi.com/quote/search"
    querystring = {'query': query, }
    if maxlength:
        querystring['maxlength'] = maxlength
    if minlength:
        querystring['minlength'] = minlength
    if category:
        querystring['category'] = category
    if author:
        querystring['author'] = author
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theysaidso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

