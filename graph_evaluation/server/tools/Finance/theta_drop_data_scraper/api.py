import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_search_results(sort_by: str=None, page: int=1, number: int=100, query: str=None, content_creator: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all search results from the THETA Drop marketplace search engine."
    query: The search query.
        content_creator: Filters the search results with a creator id
        type: Filter by the type of content you want to search: token, pack or redeemable
        
    """
    url = f"https://theta-drop-data-scraper.p.rapidapi.com/search"
    querystring = {}
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if number:
        querystring['number'] = number
    if query:
        querystring['query'] = query
    if content_creator:
        querystring['content_creator'] = content_creator
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theta-drop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_content(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back a content's summary and global transactions history based on its id."
    
    """
    url = f"https://theta-drop-data-scraper.p.rapidapi.com/contents"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theta-drop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_order(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back an order's summary and edition transactions history based on its id."
    
    """
    url = f"https://theta-drop-data-scraper.p.rapidapi.com/orders"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theta-drop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_sales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back the top sales from THETA Drop"
    
    """
    url = f"https://theta-drop-data-scraper.p.rapidapi.com/sales/top"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theta-drop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_sales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back the latest sales data from THETA Drop"
    
    """
    url = f"https://theta-drop-data-scraper.p.rapidapi.com/sales/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theta-drop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

