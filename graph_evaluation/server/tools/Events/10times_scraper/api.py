import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_categories(query: str='bio', sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Categories"
    
    """
    url = f"https://10times-scraper.p.rapidapi.com/get-categories"
    querystring = {}
    if query:
        querystring['query'] = query
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10times-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_events(category: str=None, query: str=None, sort: str=None, limit: int=None, page: int=None, location: str='Jakarta', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Events"
    category: Category ids separated by spaces or comma.  Use` get-category` endpoint to list categories.
        query: Search event name
        
    """
    url = f"https://10times-scraper.p.rapidapi.com/search-events"
    querystring = {}
    if category:
        querystring['category'] = category
    if query:
        querystring['query'] = query
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10times-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

