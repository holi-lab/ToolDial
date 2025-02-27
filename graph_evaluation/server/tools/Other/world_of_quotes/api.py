import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_quotes_by_specific_category(category: str, page: int=2, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the collection of quotes by category you pass in the query params. Maximum you can get 20 quotes at a time and can be customized by *limit*. Our API support pagination and records can be paginated by *page* query param."
    
    """
    url = f"https://world-of-quotes.p.rapidapi.com/v1/quotes"
    querystring = {'category': category, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quote_of_the_day(author: str=None, category: str='inspirational', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the handpicked quote of the day among 45,000+ quotes based on the highest ratings.
		You may also get quote of the day of specific *author* or *category*."
    
    """
    url = f"https://world-of-quotes.p.rapidapi.com/v1/quotes/quote-of-the-day"
    querystring = {}
    if author:
        querystring['author'] = author
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quotes_by_specific_author(author: str, page: int=3, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the collection of quotes by author you pass in the query params. Maximum you can get 20 quotes at a time and can be customized by *limit*. Our API support pagination and records can be paginated by *page* query param."
    
    """
    url = f"https://world-of-quotes.p.rapidapi.com/v1/quotes"
    querystring = {'author': author, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_authors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns array of over 10,000 authors we have in our library.
		This returns an array of strings."
    
    """
    url = f"https://world-of-quotes.p.rapidapi.com/v1/quotes/author"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quotes_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns array of over 350 types of categories we have in our library.
		This returns an array of strings. Real response will have more categories but in mock response we displayed a few."
    
    """
    url = f"https://world-of-quotes.p.rapidapi.com/v1/quotes/category"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

