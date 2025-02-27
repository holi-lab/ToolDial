import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def market_get_movers(is_id: str, template: str='INDEX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest information of movers in the market"
    id: The value of id field returned in .../market/auto-complete endpoint.
        template: One of the following COMMODITY|CURRENCY|INDEX|INDEXFUTURE|RATE|STOCK
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-movers"
    querystring = {'id': is_id, }
    if template:
        querystring['template'] = template
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stories_detail(internalid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full story information"
    internalid: Get suitable value from .../stories/list or .../news/list or .../news/list-by-region endpoints
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/stories/detail"
    querystring = {'internalID': internalid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by category"
    id: One of the following is allowed markets|technology|view|pursuits|politics|green|citylab|businessweek|fixed-income|hyperdrive|cryptocurrencies|wealth|latest|personalFinance|quickTake|world|industries|stocks|currencies|brexit
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/news/list"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_region(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news from different categories and grouped by region"
    id: One of the following is allowed home-v3|asia-home-v3|europe-home-v3|middle-east-home-v3|africa-home-v3|canada-home-v3
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/news/list-by-region"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_cross_currencies(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange rate between currencies"
    id: The currency code, separated by comma to query multiple currencies at once
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-cross-currencies"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_price_chart(is_id: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related data to draw price chart"
    id: The value of id field returned in .../market/auto-complete endpoint.
        interval: One of the followings : d1|d3|ytd|m1|m3|m6|y1|y5
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-price-chart"
    querystring = {'id': is_id, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_chart(is_id: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related data for drawing chart"
    id: The value of id field returned in .../market/auto-complete endpoint
        interval: One of the followings : d1|d3|ytd|m1|m3|m6|y1|y5
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-chart"
    querystring = {'id': is_id, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stories_list(is_id: str, template: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of stories related to a sticker"
    id: The value of id field returned in .../market/auto-complete endpoint
        template: One of the followings : COMMODITY|CURRENCY|INDEX|INDEXFUTURE|RATE|STOCK
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/stories/list"
    querystring = {'id': is_id, 'template': template, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_financials(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get financial information of stocks"
    id: The id of stickers, use market/auto-complete API to get correct values
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/stock/get-financials"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query suggestion by term and phrase"
    query: name of company, stock, organization, etc...
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_full(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all field and information about tickers"
    id: The value of id field returned in .../market/auto-complete endpoint, separated by comma to query multiple stickers at once.
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-full"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_compact(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most informative fields about indices, commodities, currencies, rates, etc..."
    id: The value of id field returned in .../market/auto-complete endpoint, separated by comma to query multiple stickers at once.
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/market/get-compact"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_statistics(is_id: str, template: str='STOCK', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics information of stocks"
    id: The id of stickers, use market/auto-complete API to get correct values
        template: One of the following COMMODITY|CURRENCY|INDEX|INDEXFUTURE|RATE|STOCK
        
    """
    url = f"https://bloomberg-market-and-financial-news.p.rapidapi.com/stock/get-statistics"
    querystring = {'id': is_id, }
    if template:
        querystring['template'] = template
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloomberg-market-and-financial-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

