import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dad_jokes_joke_of_the_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return joke of the day. Each new UTC day API will return a new joke. During the UTC day, API returns the same joke."
    
    """
    url = f"https://dad-jokes7.p.rapidapi.com/dad-jokes/joke-of-the-day"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dad_jokes_search(text: str='dad', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a joke by a keyword. Keyword will be used as is in the search. E.g. if you need to search for a joke with the keyword "dad" in it, you will need to pass "dad" as the keyword. If you need to search for a joke with specific word, you will need to pass keyword with surrounding spaces or punctuation, e.g. " dad". Search is not case-sensitive.
		
		API return is limited to 3 records per one search request."
    text: 'text' query parameter is required and should be more than 3 characters long.
        
    """
    url = f"https://dad-jokes7.p.rapidapi.com/dad-jokes/search"
    querystring = {}
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dad_jokes_random(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return random joke."
    
    """
    url = f"https://dad-jokes7.p.rapidapi.com/dad-jokes/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dad_jokes_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the health status of the API. Returns current UTC time."
    
    """
    url = f"https://dad-jokes7.p.rapidapi.com/dad-jokes/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

