import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Through this endpoint you'll get a list of the most popular tags, which you can use to filter quotes by using the *only_tags* or *except_tags* filters."
    
    """
    url = f"https://good-quotes2.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "good-quotes2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_single_quote_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single quote by id"
    
    """
    url = f"https://good-quotes2.p.rapidapi.com/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "good-quotes2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_quotes(max_length: int=400, only_tags: str=None, except_tags: str=None, keyword: str='love', count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Through this endpoint, you can get a single or a list of  random quotes."
    count: The number of random quotes to return. Must be between 1 - 20, defaults to 1.
        
    """
    url = f"https://good-quotes2.p.rapidapi.com/random"
    querystring = {}
    if max_length:
        querystring['max_length'] = max_length
    if only_tags:
        querystring['only_tags'] = only_tags
    if except_tags:
        querystring['except_tags'] = except_tags
    if keyword:
        querystring['keyword'] = keyword
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "good-quotes2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_quotes(page: str='1', per_page: str='20', keyword: str='life', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of paginated quotes, sorted by popularity."
    keyword: You can pass a keyword to get only quotes that contain it.
        
    """
    url = f"https://good-quotes2.p.rapidapi.com/"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "good-quotes2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

