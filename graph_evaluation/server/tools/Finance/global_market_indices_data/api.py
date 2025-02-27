import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index_last_price(index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lates price of an index"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/index_last_price"
    querystring = {'index': index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_top5_constituents(index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 5 gainers/losers of an index  over a period"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/index_top5_constituents"
    querystring = {'index': index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_historic_price(index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historic daily price of an index over a period"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/index_historic_price"
    querystring = {'index': index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available Indices"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/indices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_constituents(index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Index constituents"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/index_constituents"
    querystring = {'index': index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index_price_change(index: str, period: str='7DAYS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Index change over the period of time(1DAY, 7DAYS, 1MONTH, 3MONTHS, 6MONTHS, 1YEAR )"
    
    """
    url = f"https://global-market-indices-data.p.rapidapi.com/v1/index_price_change"
    querystring = {'index': index, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-market-indices-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

