import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_bargain_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will take a node id and return data about the specified bargain, including the product, price, whether the deal has expired, the referrer, and user that posted the deal."
    
    """
    url = f"https://ozbargain-scraper.p.rapidapi.com/bargain/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ozbargain-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category_items(category_name: str, show_expired: bool=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of bargains based on a specified category. This includes the title, description, sale price, referrer, url, and user upload details."
    
    """
    url = f"https://ozbargain-scraper.p.rapidapi.com/category/{category_name}"
    querystring = {}
    if show_expired:
        querystring['show_expired'] = show_expired
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ozbargain-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_results(query: str, show_expired: bool=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a search query and returns details on all results for that search query including title, price, link to bargain, user details, date posted, up/downvotes etc."
    
    """
    url = f"https://ozbargain-scraper.p.rapidapi.com/search/{query}"
    querystring = {}
    if show_expired:
        querystring['show_expired'] = show_expired
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ozbargain-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

