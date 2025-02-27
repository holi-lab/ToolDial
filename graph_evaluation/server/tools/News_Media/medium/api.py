import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_topics(q: str, limit: int=10, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: query to search by
        limit: Number of results to return
        page: Page offset
        
    """
    url = f"https://medium16.p.rapidapi.com/search/topics"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stories_topic(topic: str, limit: int=10, page: int=0, filter: str='latest', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    topic: Topic slug
        limit: Number of results to return
        page: Page offset
        filter: Page offset
        
    """
    url = f"https://medium16.p.rapidapi.com/stories/{topic}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_stories(q: str, page: int=0, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: query to search by
        page: Page offset
        limit: Number of results to return
        
    """
    url = f"https://medium16.p.rapidapi.com/search/stories"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

