import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stock_historical_data(company_code: str, sort: str='DESC', start_date: str='2022-10-01', page: int=1, end_date: str='2022-10-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to get Stock Historical data by Company Code"
    
    """
    url = f"https://stock-market-api-indian-stock-market.p.rapidapi.com/api/stocks/{company_code}/history"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if start_date:
        querystring['start_date'] = start_date
    if page:
        querystring['page'] = page
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_stocks(term: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to search Stocks by company name or company code"
    
    """
    url = f"https://stock-market-api-indian-stock-market.p.rapidapi.com/api/search_stocks"
    querystring = {'term': term, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stocks_list(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to get List of Stocks from BSE and NSE Stock Exchanges"
    
    """
    url = f"https://stock-market-api-indian-stock-market.p.rapidapi.com/api/stocks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-api-indian-stock-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

