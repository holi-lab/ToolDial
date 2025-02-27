import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will list all available word categories. Check this before calling /stopwords"
    
    """
    url = f"https://stopwords.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopwords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stopwords(langs: str=None, details: bool=None, categories: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return the exact extract words to use in your data analysis or application"
    langs: Comma seperated list of language isocodes
        details: Boolean (1 or 0) indication weather you want details about each word
        categories: Comma seperated list of categories. Use /categories to get ID's
        
    """
    url = f"https://stopwords.p.rapidapi.com/stopwords"
    querystring = {}
    if langs:
        querystring['langs'] = langs
    if details:
        querystring['details'] = details
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopwords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

