import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def born_on_this_day(month: str=None, day: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a on this day birthday entry"
    month: Month value. Defaults to current month
        day: Optional day of the month (1- 28/30/31 based on the month). Defaults to current day of the month.
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact/onthisday/born"
    querystring = {}
    if month:
        querystring['month'] = month
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def died_on_this_day(day: str='undefined', month: str='undefined', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a on this day death entry"
    day: Optional day of the month (1- 28/30/31 based on the month). Defaults to current day of the month.
        month: Month value. Defaults to current month
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact/onthisday/died"
    querystring = {}
    if day:
        querystring['day'] = day
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_fact(category: str=None, subcategory: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random fact"
    category: Optional category for the random fact.
        subcategory: Optional subcategory for the random fact 
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact/random"
    querystring = {}
    if category:
        querystring['category'] = category
    if subcategory:
        querystring['subcategory'] = subcategory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_facts(query: str, subcategory: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a fact related to the query text"
    query: Query text to search the facts database
        subcategory: Optional sub category to filter the search
        category: Optional category to filter the search
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact/search"
    querystring = {'query': query, }
    if subcategory:
        querystring['subcategory'] = subcategory
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fact_categories(start: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available categories in the system"
    start: Response is paged. This specifies the start value.
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact/categories"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fact(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific fact entry"
    id: ID of the fact entry
        
    """
    url = f"https://random-facts1.p.rapidapi.com/fact"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-facts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

