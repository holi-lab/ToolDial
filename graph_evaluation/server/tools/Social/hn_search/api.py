import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def items(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-hn-search.p.rapidapi.com/items/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hn-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-hn-search.p.rapidapi.com/users/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hn-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_date(tags: str='(story,poll)', query: str=None, numericfilters: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    tags: ilter on a specific tag. Available tags: story comment poll pollop show_hn ask_hn author_:USERNAME story_:ID
        query: full-text query
        numericfilters: filter on a specific numerical condition (<, <=, =, > or >). Available numerical fields: created_at_i points num_comments
        page: page number
        
    """
    url = f"https://community-hn-search.p.rapidapi.com/search_by_date"
    querystring = {}
    if tags:
        querystring['tags'] = tags
    if query:
        querystring['query'] = query
    if numericfilters:
        querystring['numericFilters'] = numericfilters
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hn-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str='foo', tags: str='story', numericfilters: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    query: full-text query
        tags: filter on a specific tag. Available tags: story comment poll pollop show_hn ask_hn author_:USERNAME story_:ID
        numericfilters: filter on a specific numerical condition (<, <=, =, > or >). Available numerical fields: created_at_i points num_comments
        page: page number
        
    """
    url = f"https://community-hn-search.p.rapidapi.com/search"
    querystring = {}
    if query:
        querystring['query'] = query
    if tags:
        querystring['tags'] = tags
    if numericfilters:
        querystring['numericFilters'] = numericfilters
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-hn-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

