import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_price_and_details_by_country(cc: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more info about a game such as:
		- description
		- price overview
		- package groups with names and prices
		- platform
		- categories
		- metacritic score
		- achievements"
    cc: Country code in lowercase. [List here](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes)
        id: To get the id you should use GET /search
        
    """
    url = f"https://game-price.p.rapidapi.com/v1/appDetailsByCountry"
    querystring = {'cc': cc, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_price_and_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more info about a game such as:
		- description
		- price overview
		- package groups with names and prices
		- platform
		- categories
		- metacritic score
		- achievements"
    id: To get the id you should use GET /search
        
    """
    url = f"https://game-price.p.rapidapi.com/v1/appDetails"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_price_by_country(cc: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game price from different countries"
    cc: Country code in lowercase. [List here](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes)
        
    """
    url = f"https://game-price.p.rapidapi.com/v1/priceByCountry"
    querystring = {'cc': cc, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_price(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current price and discount"
    id: To get the id you should use GET /search
        
    """
    url = f"https://game-price.p.rapidapi.com/v1/price"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for games in steam and get basic information about them"
    
    """
    url = f"https://game-price.p.rapidapi.com/v1/search"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

