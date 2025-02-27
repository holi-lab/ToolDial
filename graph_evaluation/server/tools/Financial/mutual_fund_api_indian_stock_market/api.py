import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mutual_fund_historical_data(schema_code: str, start_date: str='2021-11-10', sort: str='ASC', end_date: str='2021-12-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Historical data of Mutual Funds. Supporting Start date and End date"
    sort: Sort by Schema Code
        
    """
    url = f"https://mutual-fund-api-indian-stock-market.p.rapidapi.com/api/mutual_funds/{schema_code}/history"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if sort:
        querystring['sort'] = sort
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mutual-fund-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_mutual_funds(term: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Mutual funds by Name"
    
    """
    url = f"https://mutual-fund-api-indian-stock-market.p.rapidapi.com/api/search_mutual_funds"
    querystring = {'term': term, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mutual-fund-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_mutual_funds_list(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of all Mutual funds with their latest NAV"
    
    """
    url = f"https://mutual-fund-api-indian-stock-market.p.rapidapi.com/api/mutual_funds"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mutual-fund-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

