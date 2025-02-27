import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_trending_india_companies_stocks_by_fundamental(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TopTrending Indian companies stock live data with respective to Fundamental"
    
    """
    url = f"https://india-trending-stock-api.p.rapidapi.com/india_trending_stocks_by_fundamental"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "india-trending-stock-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_trending_india_companies_stocks_by_technical(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TopTrending Indian companies stock live data with respective to Technical"
    
    """
    url = f"https://india-trending-stock-api.p.rapidapi.com/india_trending_stocks_by_technical"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "india-trending-stock-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_trending_india_companies_stocks_by_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get TopTrending Indian companies stock live data with respective to Performance"
    
    """
    url = f"https://india-trending-stock-api.p.rapidapi.com/india_trending_stocks_by_performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "india-trending-stock-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_india_companies_trending_stocks_by_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Top Indian companies stock live data with respective price"
    
    """
    url = f"https://india-trending-stock-api.p.rapidapi.com/india_trending_stocks_by_price"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "india-trending-stock-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

