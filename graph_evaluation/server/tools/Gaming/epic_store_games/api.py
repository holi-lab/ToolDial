import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_coming_soon_games(searchwords: str, categories: str=None, country: str='us', locale: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Coming Soon Games"
    categories: Default: **Games**
        country: Default: **us** --- *For the price*
        locale: Default: **us** --- *For the language*
        
    """
    url = f"https://epic-store-games.p.rapidapi.com/comingSoon"
    querystring = {'searchWords': searchwords, }
    if categories:
        querystring['categories'] = categories
    if country:
        querystring['country'] = country
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epic-store-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_on_sale_games(searchwords: str, locale: str='us', country: str='us', categories: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search On Sale Games"
    locale: Default: **us** --- *For the language*
        country: Default: **us** --- *For the price*
        categories: Default: **Games**
        
    """
    url = f"https://epic-store-games.p.rapidapi.com/onSale"
    querystring = {'searchWords': searchwords, }
    if locale:
        querystring['locale'] = locale
    if country:
        querystring['country'] = country
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epic-store-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

