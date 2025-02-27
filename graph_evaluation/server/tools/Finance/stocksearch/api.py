import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company_profile(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a company profile containing information such as a description, sector, address, etc."
    
    """
    url = f"https://stocksearch.p.rapidapi.com/api/stocks/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocksearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_stocks_v2(query: str, searchmetadata: bool=None, offset: int=0, size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search stocks by name, symbol, or related item."
    query: Search keywords
        searchmetadata: Search the company's description or related items
        offset: Pagination offset from the previous set of results. For example, if `offset=5`, the results set will start at the 6th item from the total results. Note, 200 is the max number of results that will be returned in total.
        size: Max number of results to get in one response. (Max value: 20)
        
    """
    url = f"https://stocksearch.p.rapidapi.com/api/v2/stocks"
    querystring = {'query': query, }
    if searchmetadata:
        querystring['searchMetadata'] = searchmetadata
    if offset:
        querystring['offset'] = offset
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocksearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_stocks(query: str, size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search stocks by name or symbol"
    query: Full or partial ticker symbol or company name
        size: Number of results to return (max: 20)
        
    """
    url = f"https://stocksearch.p.rapidapi.com/api/stocks"
    querystring = {'query': query, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocksearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

