import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_bingsearch_get(query: str, num: int=10, region: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    num: Maximum quantity of links to return, valid values are integers: options (10,15,30,50)
        region: The region parameter value is two-letter language with a dash and two-letter country code. Search results from specific region of origin. Example: en-us
        
    """
    url = f"https://youtube-search12.p.rapidapi.com/bingSearch"
    querystring = {'query': query, }
    if num:
        querystring['num'] = num
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_yandexsearch_get(query: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    language: The language parameter value is two-letter language code
        
    """
    url = f"https://youtube-search12.p.rapidapi.com/yandexSearch"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

