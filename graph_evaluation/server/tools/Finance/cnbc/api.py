import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def symbols_translate(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get issueId from specific symbol"
    symbol: Symbol of stock, index, exchange, etc...
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/translate"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_peers(symbol: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get peers relating to stock quote, index, exchange, etc..."
    symbol: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. 
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-peers"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_priceline_chart(issueid: int, numberofdays: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate image of price line chart of specific stock quote, index, exchange, etc..."
    issueid: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. 
        numberofdays: Number of days to generate image of price line chart (1 to 9999)
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-priceline-chart"
    querystring = {'issueId': issueid, }
    if numberofdays:
        querystring['numberOfDays'] = numberofdays
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_summary(issueids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information of stock quote, index, exchange, etc..."
    issueids: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. Separated by comma for multiple values. Ex : 36276,24812378
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-summary"
    querystring = {'issueIds': issueids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_fundamentals(issueids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fundamental information of stock quote, index, exchange, etc..."
    issueids: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. Separated by comma for multiple values. Ex : 36276,24812378
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-fundamentals"
    querystring = {'issueIds': issueids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_earnings_chart(issueid: int, numberofyears: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate image of earnings chart of specific stock quote, index, exchange, etc..."
    issueid: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. 
        numberofyears: Number of most latest years to get report of earnings (1 to 10)
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-earnings-chart"
    querystring = {'issueId': issueid, }
    if numberofyears:
        querystring['numberOfYears'] = numberofyears
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_v2_get_chart(symbol: str, interval: str='1D', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get raw data to draw price line chart of stock quote, index, exchange, etc..."
    symbol: The value of 'symbolName' returned in .../auto-complete OR 'symbol' returned in .../v2/auto-complete endpoint.
        interval: One of the following is allowed 1D|5D|1M|6M|YTD|1Y|5Y|ALL
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/v2/get-chart"
    querystring = {'symbol': symbol, }
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_by_symbol(symbol: str, page: int=1, pagesize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news by symbol name"
    symbol: The symbol returned in .../v2/auto-complete or other endpoints
        page: For paging purpose
        pagesize: For paging purpose
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/v2/list-by-symbol"
    querystring = {'symbol': symbol, }
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list(franchiseid: int, count: int=15, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by category"
    franchiseid: The value of .../sectionData/parameters/franchiseId json object returned in .../get-meta-data endpoint. Pass this parameter multiple times for news from multiple categories. Ex : ...&franchiseId=105230142&franchiseId=15839263&...
        count: The number of items returned by the endpoint
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/v2/list"
    querystring = {'franchiseId': franchiseid, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_special_reports(pagesize: int=30, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List special reports"
    pagesize: For paging purpose
        page: For paging purpose
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/v2/list-special-reports"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_trending(tag: str, count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending news"
    tag: The value is always Articles
        count: Number of items returned by the endpoint
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/v2/list-trending"
    querystring = {'tag': tag, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_profile(issueid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information of stock quote, index, exchange, etc..."
    issueid: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints. 
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-profile"
    querystring = {'issueId': issueid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_chart_deprecating(symbol: int, interval: str='1d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get raw data to draw price line chart of stock quote, index, exchange, etc..."
    symbol: The value of issueId field returned in .../auto-complete or .../symbols/translate endpoints
        interval: One of the following is allowed 1d|5d|1m|3m|6m|ytd|1y|3y|5y|10y|all
        
    """
    url = f"https://cnbc.p.rapidapi.com/symbols/get-chart"
    querystring = {'symbol': symbol, }
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto suggestion by familiar terms or phrase"
    q: Any word or phrase that you are familiar with
        
    """
    url = f"https://cnbc.p.rapidapi.com/v2/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meta_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meta data that supports for other endpoints"
    
    """
    url = f"https://cnbc.p.rapidapi.com/get-meta-data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_trending_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending news"
    
    """
    url = f"https://cnbc.p.rapidapi.com/news/list-trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_special_reports_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List special reports"
    
    """
    url = f"https://cnbc.p.rapidapi.com/news/list-special-reports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete_deprecated(prefix: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto suggestion by familiar words or phrase"
    prefix: Any word or phrase that you are familiar with
        
    """
    url = f"https://cnbc.p.rapidapi.com/auto-complete"
    querystring = {'prefix': prefix, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_deprecated(franchiseid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by category"
    franchiseid: The value of .../sectionData/parameters/franchiseId json object returned in .../get-meta-data endpoint
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/list"
    querystring = {'franchiseId': franchiseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_symbol_deprecated(tickersymbol: str, page: int=1, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news by symbol name"
    tickersymbol: The value of symbolName field returned in .../auto-complete endpoint
        page: For paging purpose
        pagesize: For paging purpose
        
    """
    url = f"https://cnbc.p.rapidapi.com/news/list-by-symbol"
    querystring = {'tickersymbol': tickersymbol, }
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pagesize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_list_indices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available indices"
    
    """
    url = f"https://cnbc.p.rapidapi.com/market/list-indices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cnbc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

