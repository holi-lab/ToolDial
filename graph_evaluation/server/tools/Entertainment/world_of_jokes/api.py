import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_joke_of_the_day_by_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the joke of the day of specific category from a collection of most rated and most popular jokes."
    category: Category of joke based on the jokes categories API
        
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/joke-of-the-day-by-category"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_joke_by_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the random joke by category from a collection of most rated and most popular jokes."
    category: Category of joke based on the jokes categories API
        
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/random-joke-by-category"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_joke(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the random joke from a collection of most rated and most popular jokes."
    
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/random-joke"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_jokes(limit: int, page: int, sortby: str='score:desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access our huge collection of jokes and paginate through them based on your desired limit and sorting criteria."
    sortby: Valid format to sort is `field:order`
e.g. `score:desc` for highest score first sorting

where `asc` for sorting in ascending order
`desc` for sorting in descending order
        
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes"
    querystring = {'limit': limit, 'page': page, }
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories_of_jokes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available categories of our Jokes collection which can be used to filter jokes based on specific category."
    
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_jokes_by_specific_category(limit: int, page: int, category: str, sortby: str='score:desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access our huge collection of jokes of specific category and paginate through them based on your desired limit and sorting criteria."
    category: Category of joke based on the jokes categories API
        sortby: Valid format to sort is `field:order`
e.g. `score:desc` for highest score first sorting
where `asc` for sorting in ascending order
`desc` for sorting in descending order
        
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/jokes-by-category"
    querystring = {'limit': limit, 'page': page, 'category': category, }
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_joke_of_the_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the joke of the day from a collection of most rated and most popular jokes."
    
    """
    url = f"https://world-of-jokes1.p.rapidapi.com/v1/jokes/joke-of-the-day"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-of-jokes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

