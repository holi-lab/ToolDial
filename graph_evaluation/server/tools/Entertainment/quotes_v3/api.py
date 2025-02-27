import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_a_random_quote(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random Quote."
    
    """
    url = f"https://orthosie-quotes-v1.p.rapidapi.com/quote/random"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orthosie-quotes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quote(query: str, maxlength: int=500, category: str=None, minlength: int=50, language: str='en', author: str='Dale Patridge', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search based on author or category and get a random quote."
    
    """
    url = f"https://orthosie-quotes-v1.p.rapidapi.com/quote/search"
    querystring = {'query': query, }
    if maxlength:
        querystring['maxlength'] = maxlength
    if category:
        querystring['category'] = category
    if minlength:
        querystring['minlength'] = minlength
    if language:
        querystring['language'] = language
    if author:
        querystring['author'] = author
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orthosie-quotes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

