import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_business_news_stream(limit: str, since: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Business news stream"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/stream/business{limit}{since}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_link_metadata(link: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get link metadata"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/link/metadata{link}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_technology_news_stream(limit: str, since: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Technology news stream"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/stream/technology{limit}{since}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_market_data(days: str, market: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market data"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v2/market/data{days}{market}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_market_news(startday: str, endday: str, max: str, offset: str, market: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market news"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/market/news{startday}{endday}{max}{offset}{market}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_portfolio_data(startday: str, endday: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get portfolio data"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/portfolio/data{startday}{endday}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_link_sentiment(link: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get link sentiment"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/link/sentiment{link}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_symbol_from_portfolio(symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove symbol from portfolio"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v2/portfolio/remove{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_news_data(startday: str, endday: str, symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily news data"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v2/symbol/mentions{startday}{endday}{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_symbol_details(symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbol details"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v2/symbol{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def add_symbol_to_portfolio(symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add symbol to portfolio"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v2/portfolio/add{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_news(startday: str, endday: str, max: str, offset: str, symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company news"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/headlines{startday}{endday}{max}{offset}{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_portfolio_news(startday: str, endday: str, max: str, offset: str, sentiment: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get portfolio news"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/portfolio/news{startday}{endday}{max}{offset}{sentiment}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_link_tags(link: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get link tags"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/link/tags{link}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_competitor_list(symbol: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitor list"
    
    """
    url = f"https://monitr-monitr-financial-data-v1.p.rapidapi.com/api/v1/competitors{symbol}{apikey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monitr-monitr-financial-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

